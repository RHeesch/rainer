import pandas as pd

integrate_ml = False        # enables the integration of ML-models

approx = True               # enables the approximation of the results
#this is necessary, if small ML models are integrated, due to their inaccuracy 

# Table of fix parameters
Parameter_fix = ["volume", "density", "specific_heat_capacity"]
#Values_fix = pd.Series([1.5, 7.870, 4.77], index = Parameter_fix) #Stahl
#Values_fix = pd.Series([1.5, 2.700, 8.96], index = Parameter_fix) #Alu
Values_fix = pd.Series([1.5, 8.730, 3.77], index = Parameter_fix) #Messing
para_fix = pd.concat([Values_fix], axis = 1)
limits_fix = ['Values_fix']
para_fix.columns = limits_fix

# Table of variable parameters
Parameter_var = ["energy_input"]
Top_var = pd.Series([1000000], index = Parameter_var)
Down_var = pd.Series([0], index = Parameter_var)
para_var = pd.concat([Top_var, Down_var], axis = 1)
limits_var = ["Top_var", "Down_var"]
para_var.columns = limits_var

Variables = ["coloured", "temperature", "milled", "drilled", "rounded"]
sym_actions = ["Mill", "Drill", "CNC", "Paint", "Heat"]
learn_actions = ["L_Heat"]

# Table of symbolic effects
Mill_eff = pd.Series(["0.0", "0.0", "10", "0.0", "0.0"], index=Variables)
Drill_eff = pd.Series(["0", "0", "0", "10", "0"], index=Variables)
CNC_eff = pd.Series(["0", "0", "0", "0", "10"], index=Variables)
Paint_eff = pd.Series(["10", "0", "0", "0", "0"], index=Variables)
Heat_eff = pd.Series(["0", "(energy_input*1000000)*((volume*density*specific_heat_capacity*1000*100)**(-1))", "0", "0", "0"], index=Variables)
effects = pd.concat([Mill_eff, Drill_eff, CNC_eff, Paint_eff, Heat_eff], axis = 1)
effects.columns = sym_actions

if integrate_ml == True:
    #Table of models
    learned_models = ["L_Heat"]
    trained_model = pd.Series(["model(state_in)"], index = learned_models)
    ld_models = pd.concat([trained_model], axis = 1)
    ld_models.columns = learned_models

    # Table of precondition of learned actions
    L_Heat_pre = pd.Series(["", "<=100", "", "", ""], index=Variables)
    L_precons = pd.concat([L_Heat_pre], axis = 1)
    L_precons.columns = learn_actions

# Table of preconditions of symbolic actions
Mill_pre = pd.Series(["0.0", "", "0.0", "", ""], index=Variables)
Drill_pre = pd.Series(["", "", "", "0.0", "0.0"], index=Variables)
CNC_pre = pd.Series(["", "", "", "", "0.0"], index=Variables)
Paint_pre = pd.Series(["0.0", ">=20", "", "", ""], index=Variables)
Heat_pre = pd.Series(["<=10", "<=100.0", "", "", ""], index=Variables)
precons = pd.concat([Mill_pre, Drill_pre, CNC_pre, Paint_pre, Heat_pre], axis = 1)
precons.columns = sym_actions

# Table of initial- & goalstate
Init = pd.Series(["0.0", "5.0", "0.0", "0.0", "0.0"], index=Variables)
Goal = pd.Series(["10", "20.0", "10", "10", "10"], index=Variables)
start_end = pd.concat([Init, Goal], axis = 1)

def print_all():
    print(para_fix)
    print(para_var)
    print(precons)
    print(effects)
    print(start_end)

print_all()

