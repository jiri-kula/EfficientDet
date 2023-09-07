# %%
import pandas as pd

meta_train = "/mnt/c/Edwards/rv5/Output/ruka_hala_6D_aug.csv"

column_names = [
    "PURPOSE",
    "PATH",
    "OBJECT",
    "X1",
    "Y1",
    "UNUSED1",
    "UNUSED2",
    "X2",
    "Y2",
    "R11",
    "R21",
    "R31",
    "R12",
    "R22",
    "R32",
]

df = pd.read_csv(meta_train, header=None, names=column_names)

# %%
# drazky, které jsou špatně anotované jako osa x směrem ke kameře
drazka = df["OBJECT"].str.contains("ne") == False
nedrazka = ~drazka

drazkou_od_kamery = df["R31"].gt(0.0)
drazkou_do_kamery = ~drazkou_od_kamery

ok_drazka = df.loc[drazka & drazkou_do_kamery]
ng_drazka = df.loc[drazka & drazkou_od_kamery]

ok_nedrazka = df.loc[nedrazka & drazkou_od_kamery]
ng_nedrazka = df.loc[nedrazka & drazkou_do_kamery]

assert len(ng_drazka) == 0
assert len(ng_nedrazka) == 0

# %%

# filter all nedrazka where x axis point towards the camera z-axis (which is wrong)
# class_is_nedrazka = df["OBJECT"].str.contains("ne")
# rotation_is_drazka = df["R31"].lt(0.0)

# wrong_nedrazka = class_is_nedrazka & rotation_is_drazka
# wrong_drazka = ~(class_is_nedrazka | rotation_is_drazka)
# wrong_x_axis = df.loc[wrong_nedrazka | wrong_drazka]


# # filter all drazka
# dr = df.loc[df["OBJECT"].str.contains("ne") == False]

# wrong_x_axis = ne.loc[ne["R31"].lt(0.0) == False]
# %%
