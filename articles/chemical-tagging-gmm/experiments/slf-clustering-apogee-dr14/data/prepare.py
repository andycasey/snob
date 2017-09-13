from astropy.table import Table

output_filename = "apogee-dr14-cannon.fits"

cannon = Table.read("allStarCannon-l31c.2.fits")
aspcap = Table.read("allStar-l31c.2.fits")

# Some basic checks.
cannon["APOGEE_ID"] = [each.strip() for each in cannon["APOGEE_ID"]]
aspcap["APOGEE_ID"] = [each.strip() for each in aspcap["APOGEE_ID"]]

assert len(cannon) == len(aspcap)
assert all(aspcap["APOGEE_ID"] == cannon["APOGEE_ID"])

# Transfer the following labels from ASPCAP table to The Cannon table:
transfer_columns = (
    "APSTAR_ID",
    "TARGET_ID",
    "ASPCAP_ID",
    "FILE",
    "APOGEE_ID",
    "TELESCOPE",
    "LOCATION_ID",
    "FIELD",
    "RA",
    "DEC",
    "GLON",
    "GLAT",
    "APOGEE_TARGET1",
    "APOGEE_TARGET2",
    "APOGEE_TARGET3",
    "TARGFLAGS",
    "SURVEY",
    "NVISITS",
    "COMBTYPE",
    "COMMISS",
    "SNR",
    "STARFLAG",
    "STARFLAGS",
    "ANDFLAG",
    "ANDFLAGS",
    "VHELIO_AVG",
    "VSCATTER",
    "VERR",
    "VERR_MED",
    "OBSVHELIO_AVG",
    "OBSVSCATTER",
    "OBSVERR",
    "OBSVERR_MED",
    "SYNTHVHELIO_AVG",
    "SYNTHVSCATTER",
    "SYNTHVERR",
    "SYNTHVERR_MED",
    "SYNTHSCATTER",
    "STABLERV_CHI2",
    "STABLERV_RCHI2",
    "CHI2_THRESHOLD",
    "STABLERV_CHI2_PROB",
    "APOGEE2_TARGET1",
    "APOGEE2_TARGET2",
    "APOGEE2_TARGET3",
    "MEANFIB",
    "SIGFIB",
    "SNREV",
    "APSTAR_VERSION",
    "ASPCAP_VERSION",
    "RESULTS_VERSION",
    "EXTRATARG",
    "VMICRO",
    "VMACRO",
    "VSINI",
    "ASPCAPFLAG",
    "ASPCAPFLAGS",
    "REDUCTION_ID",
    "PMRA",
    "PMDEC",
    "PM_SRC",
    "ALL_VISITS",
    "VISITS")

for column in transfer_columns:
    cannon[column] = aspcap[column]

# Remove rows that we don't have results for.
keep = cannon["TEFF"] > 0
cannon = cannon[keep]

cannon.write(output_filename, overwrite=True)
print("Written to {}".format(output_filename))