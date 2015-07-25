for %%p in (1,1.01,1.02,1.03,1.04,1.05,1.1,1.15,1.2,1.25,1.35,1.4,1.45,1.5,1.55,1.6,1.8,2) do (

  start python MixtureFit.py -v -p Simulation -s %%p -name %%p-
)