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
    "J",
    "J_ERR",
    "H",
    "H_ERR",
    "K",
    "K_ERR",
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
    "RV_TEFF",
    "RV_LOGG",
    "RV_FEH",
    "RV_ALPHA",
    "RV_CARB",
    "RV_CCFWHM",
    "RV_AUTOFWHM",
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
    "MIN_H",
    "MAX_H",
    "MIN_JK",
    "MAX_JK",
    "VMICRO",
    "VMACRO",
    "VSINI",
    "ASPCAPFLAG",
    "ASPCAPFLAGS",
    "REDUCTION_ID",
    "SRC_H",
    "WASH_M",
    "WASH_M_ERR",
    "WASH_T2",
    "WASH_T2_ERR",
    "DDO51",
    "DDO51_ERR",
    "IRAC_3_6",
    "IRAC_3_6_ERR",
    "IRAC_4_5",
    "IRAC_4_5_ERR",
    "IRAC_5_8",
    "IRAC_5_8_ERR",
    "IRAC_8_0",
    "IRAC_8_0_ERR",
    "WISE_4_5",
    "WISE_4_5_ERR",
    "TARG_4_5",
    "TARG_4_5_ERR",
    "AK_TARG",
    "AK_TARG_METHOD",
    "AK_WISE",
    "SFD_EBV",
    "WASH_DDO51_GIANT_FLAG",
    "WASH_DDO51_STAR_FLAG",
    "PMRA",
    "PMDEC",
    "PM_SRC",
    "ALL_VISITS",
    "VISITS")

for column in transfer_columns:
    cannon[column] = aspcap[column]

# Put NaN's into some columns?


cannon.write(output_filename, overwrite=True)
print("Written to {}".format(output_filename))