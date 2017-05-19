
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.io import fits


# In[2]:

data = Table.read("allStarCannon-l31c.2.fits")
apogee = fits.open("allStar-l31c.2.fits")


# In[3]:

apogee = apogee[1].data
print(apogee.dtype.names)


# In[4]:

from collections import Counter
fields = np.array([each.strip() for each in apogee["FIELD"]])
counter_fields = Counter(fields)

print("\n".join(sorted(set(fields))))

cluster_candidate_fields = {}
for k, v in counter_fields.items():
    if k.startswith(("M", "N", "P")):
        cluster_candidate_fields[k] = v
        print(k, v)

print("Sum: {}".format(sum(cluster_candidate_fields.values())))


# In[5]:

bad = data["TEFF"] < 0
for k in ("TEFF", "LOGG", "FE_H"):
    data[k][bad] = np.nan

# Plot and cull.
for cluster_name, num_candidates in cluster_candidate_fields.items():
    candidate = (fields == cluster_name) * np.isfinite(data["TEFF"])
    if candidate.sum() < 2:
        continue


    fig, axes = plt.subplots(1, 2)

    axes[0].set_title(cluster_name)
    scat = axes[0].scatter(data["TEFF"][candidate], data["LOGG"][candidate], c=data["FE_H"][candidate],
                          alpha=0.5)

    axes[0].set_xlim(6000, 3000)
    axes[0].set_ylim(4.5, 0)

    axes[0].text(0.05, 0.95, "{} candidates".format(candidate.sum()), transform=axes[0].transAxes)

    axes[1].scatter(apogee["VHELIO_AVG"][candidate], data["FE_H"][candidate], alpha=0.5, s=5)
    keep = np.abs(apogee["VHELIO_AVG"][candidate]) < 500
    _ = apogee["VHELIO_AVG"][candidate][keep]
    axes[1].set_xlim(_.min() - 10, _.max() + 10)
    cbar = plt.colorbar(scat)


# In[6]:

# If the cluster is in Meszaros et al. (2013), then:
# (1) Take all their members
# (2) If there are other stars, take within +/- 1\sigma of their thing.

# If it's not, then can we come up with some criteria?
meszaros = fits.open("Meszaros_et_al_2013.fits")[1].data

common_clusters = set(meszaros["Cluster"]).intersection(cluster_candidate_fields.keys())
print("Common clusters: {}".format(common_clusters))

not_common = set(meszaros["Cluster"]).difference(cluster_candidate_fields.keys())
print("Not common: {}".format(not_common))
# M35 and N2158 have no finite teffs.


# In[7]:

# OK, Take all the Meszaros members, and stars that are within the +/- tolerance of the VHELIO_AVG?

default_feh_tolerance = 0.50 # dex
default_vhelio_tolerance = 10.0 # km/s
vhelio_tolerances = {"N7789": 5, "N6791": 5, "M13": 5, "N188": 5, }
feh_tolerances = {"M67": 0.10, "N7789": 0.10, "M71": 0.20, "M92": 0.25, "N188": 0.10}

match_meszaros_member = np.zeros(len(data), dtype=bool)
match_likely_member = np.zeros(len(data), dtype=bool)

ignore_clusters = ("Pleiades", )

for cluster_name in set(common_clusters).difference(ignore_clusters):
    candidate = (fields == cluster_name) * np.isfinite(data["TEFF"])

    member_names = meszaros["_2MASS"][meszaros["Cluster"] == cluster_name]
    candidate_is_member = np.array([_[2:] in member_names for _ in data["APOGEE_ID"][candidate]])

    meszaros_member = np.zeros(candidate.shape, dtype=bool)
    meszaros_member[candidate] = candidate_is_member

    #assert candidate_is_member.sum() >= len(member_names)

    print("{} candidates: {}".format(cluster_name, candidate.sum()))
    print("{} Mezaros members: {}".format(cluster_name, meszaros_member.sum()))

    median_feh_members = np.median(data["FE_H"][meszaros_member])
    median_vhelio_members = np.median(apogee["VHELIO_AVG"][meszaros_member])

    feh_tolerance = feh_tolerances.get(cluster_name, default_feh_tolerance)
    vhelio_tolerance = vhelio_tolerances.get(cluster_name, default_vhelio_tolerance)
    likely_members = candidate                    * (np.abs(apogee["VHELIO_AVG"] - median_vhelio_members) < vhelio_tolerance)                    * (np.abs(data["FE_H"] - median_feh_members) < feh_tolerance)

    match_meszaros_member += meszaros_member
    match_likely_member += likely_members

    fig, axes = plt.subplots(1, 2)

    axes[0].set_title(cluster_name)
    scat = axes[0].scatter(data["TEFF"][candidate], data["LOGG"][candidate],
                           facecolor="#666666", alpha=0.1)

    _ = data["FE_H"][likely_members]
    if likely_members.sum() > 0:
        vmin, vmax = (min(_), max(_))
        scat = axes[0].scatter(data["TEFF"][likely_members], data["LOGG"][likely_members],
                               c=data["FE_H"][likely_members], alpha=0.5, vmin=vmin, vmax=vmax)
        _ = axes[0].scatter(data["TEFF"][meszaros_member], data["LOGG"][meszaros_member],
                            c=data["FE_H"][meszaros_member], vmin=vmin, vmax=vmax,
                            edgecolor="k")
        cbar = plt.colorbar(scat)

    axes[0].set_xlim(6000, 3000)
    axes[0].set_ylim(4.5, 0)

    axes[0].text(0.05, 0.95, "{} candidates".format(candidate.sum()), transform=axes[0].transAxes)
    axes[0].text(0.05, 0.80, "{} Meszaros".format(meszaros_member.sum()), transform=axes[0].transAxes)
    axes[0].text(0.05, 0.65, "{} likely members".format(likely_members.sum()), transform=axes[0].transAxes)

    axes[1].scatter(apogee["VHELIO_AVG"][candidate], data["FE_H"][candidate],
                   facecolor="#666666", alpha=0.5, s=5)
    axes[1].scatter(apogee["VHELIO_AVG"][likely_members], data["FE_H"][likely_members],
                   facecolor="b", alpha=0.75, s=5)

    if likely_members.sum() > 0:
        _ = apogee["VHELIO_AVG"][likely_members]
        axes[1].set_xlim(_.min() - 100, _.max() + 100)


# In[8]:

print("Meszaros members: {}".format(match_meszaros_member.sum()))
print("Likely members: {}".format(match_likely_member.sum()))

distilled_cluster_names = list(set(apogee["FIELD"][match_meszaros_member]))
print("Number of clusters: {}".format(len(distilled_cluster_names), distilled_cluster_names))

# [Fe/H] distributions?
fig, ax = plt.subplots()
bins = np.arange(-3, 0.6, 0.10)
_ = ax.hist([
        data["FE_H"][match_meszaros_member],
        data["FE_H"][match_likely_member]
    ], bins=bins, label=("Meszaros", "Likely"))
plt.legend()


# In[11]:

# Join the two tables together, and provide indicator variables for membership in clusters.
column_names = (
 'APSTAR_ID',
 'TARGET_ID',
 'ASPCAP_ID',
 'FILE',
 'APOGEE_ID',
 'TELESCOPE',
 'LOCATION_ID',
 'FIELD',
 'J',
 'J_ERR',
 'H',
 'H_ERR',
 'K',
 'K_ERR',
 'RA',
 'DEC',
 'GLON',
 'GLAT',
 'APOGEE_TARGET1',
 'APOGEE_TARGET2',
 'APOGEE_TARGET3',
 'TARGFLAGS',
 'SURVEY',
 'NINST',
 'NVISITS',
 'COMBTYPE',
 'COMMISS',
 'SNR',
 'STARFLAG',
 'STARFLAGS',
 'ANDFLAG',
 'ANDFLAGS',
 'VHELIO_AVG',
 'VSCATTER',
 'VERR',
 'VERR_MED',
 'OBSVHELIO_AVG',
 'OBSVSCATTER',
 'OBSVERR',
 'OBSVERR_MED',
 'SYNTHVHELIO_AVG',
 'SYNTHVSCATTER',
 'SYNTHVERR',
 'SYNTHVERR_MED',
 'RV_TEFF',
 'RV_LOGG',
 'RV_FEH',
 'RV_ALPHA',
 'RV_CARB',
 'RV_CCFWHM',
 'RV_AUTOFWHM',
 'SYNTHSCATTER',
 'STABLERV_CHI2',
 'STABLERV_RCHI2',
 'CHI2_THRESHOLD',
 'STABLERV_CHI2_PROB',
 'APOGEE2_TARGET1',
 'APOGEE2_TARGET2',
 'APOGEE2_TARGET3',
 'MEANFIB',
 'SIGFIB',
 'SNREV',
 'APSTAR_VERSION',
 'AK_TARG',
 'AK_TARG_METHOD',
 'AK_WISE',
 'SFD_EBV',
 'WASH_DDO51_GIANT_FLAG',
 'WASH_DDO51_STAR_FLAG',
 'PMRA',
 'PMDEC',
 'PM_SRC',
 'ALL_VISITS',
 'VISITS',
 'ALL_VISIT_PK',
 'VISIT_PK',
 'FPARAM_CLASS',
 'CHI2_CLASS'
 )
for column_name in column_names:
    data[column_name] = apogee[column_name].copy()

# Provide indicator variables.
membership_probability = np.zeros(len(data), dtype=float)
membership_probability[match_likely_member] = 0.5
membership_probability[match_meszaros_member] = 1.0

associations = data["FIELD"].copy()
associations[membership_probability == 0.0] = ""

data["ASSOCIATION_PROB"] = np.array(membership_probability)


# In[12]:

# [Fe/H] and [Mg/H]
fig, axes = plt.subplots(1, 2)
meszaros_indices = [distilled_cluster_names.index(field) for field in apogee["FIELD"][match_meszaros_member]]
likely_indices = [distilled_cluster_names.index(field) for field in apogee["FIELD"][match_likely_member]]
axes[0].scatter(
    data["FE_H"][match_meszaros_member], data["MG_H"][match_meszaros_member],
    c=meszaros_indices)
axes[0].set_title("Meszaros members")
axes[1].scatter(
    data["FE_H"][match_likely_member], data["MG_H"][match_likely_member],
    c=likely_indices, alpha=0.5)
axes[1].set_title("Likely members")

elements = [_ for _ in data.dtype.names if _.endswith("_H")]
print("Available elemental abundances ({}):\n{}".format(len(elements), ", ".join(elements)))


# In[ ]:

# For saving:
plt.close("all")
del data["FILENAME"]
del apogee
data.write("../apogee-dr14-catalog.fits.gz")

#data.write("../apogee-dr14-catalog.fits")


# In[ ]:



