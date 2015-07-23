for %%p in (1,1.01,1.02,1.03,1.04,1.05,1.1,1.15,1.2,1.25) do (

  start python MixtureFit.py -v -p Simulation -s %%p -name %%p-
)