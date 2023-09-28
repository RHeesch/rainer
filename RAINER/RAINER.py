import numpy as np
import pandas as pd
import math as math
#import translateSMT_FliPSi as ge    #add, if an ML model is integrated
from tqdm import tqdm, trange
import torch
import z3
import copy
import numpy as np
import input_FliPSi as ip
from datetime import datetime

integrate_learned_model = ip.integrate_ml

start_ex = datetime.now()

index = 0                   # stepcounter
max_step_number = 5         # maximum number of steps
plan = []                   # the plan leading from the intial to the goal state

precons_colums = list(ip.precons.columns.values)
precons_rows = list(ip.precons.index.values)
effect_colums = list(ip.effects.columns.values)
effect_rows = list(ip.effects.index.values)
start_end_colums = list(ip.start_end.columns.values)
start_end_rows = list(ip.start_end.index.values)
if integrate_learned_model == True:
    L_precons_colums = list(ip.L_precons.columns.values)
    L_precons_rows = list(ip.L_precons.index.values)
    ld_models_colums = list(ip.ld_models.columns.values)
    ld_models_rows = list(ip.ld_models.index.values)
    all_effects = effect_colums + ld_models_colums
else:
    all_effects = effect_colums

# test for valid input
if len(start_end_colums) != 2:
    print('The Defintion of inital and goal state is not correct.')
    exit()
for i in range(len(precons_colums)):
    for l in range(len(precons_colums[i])):
        if precons_colums[i][l] != effect_colums[i][l]:
            print('Tables of precons and effects do not match.')
            exit()
for i in range(len(precons_rows)):
    for l in range(len(precons_rows[i])):
        if precons_rows[i][l] != effect_rows[i][l]:
            print('Tables of precons and effects do not match.')
            exit()

# Build lists of fix parameters
para_fix_colums = list(ip.para_fix.columns.values)
para_fix_rows = list(ip.para_fix.index.values)

# Build lists of variable parameters
para_var_colums = list(ip.para_var.columns.values)
para_var_rows = list(ip.para_var.index.values)

# Build EnumSort of Variables
list_of_var = []
variables, list_of_var = z3.EnumSort('variables', effect_rows)

# Build lists for vaules of fix parameters
list_of_fix_para = []
for i in range(len(para_fix_rows)):
    list_of_fix_para.append(para_fix_rows[i])
    list_of_fix_para[i] = z3.Real(para_fix_rows[i])

# Build lists for vaules of variable parameters
list_of_var_para = []
for i in range(len(para_var_rows)):
    list_of_var_para.append(para_var_rows[i])
    list_of_var_para[i] = ['list_of_var_%s_%d' % (para_var_rows[i], index)]

# Build lists of action instances
List_infra_actions = copy.deepcopy(list(all_effects))
for i in List_infra_actions:
    i = ['List_' + i + '_0']

# Build lists of instances of the actions
List_number_actions = copy.deepcopy(list(all_effects))
for i in List_number_actions:
     i = ['List_number_' + i + '_0']

# Build lists of instances of effects
List_infra_actions_effects = []
for i in range(len(all_effects)):
    List_infra_actions_effects.append('List_%s_eff' %all_effects[i])
    List_infra_actions_effects[i] = ['List_%s_eff_%d' % (all_effects[i],index)]

# Build lists of instances of precondintions
List_infra_actions_pre = []
for i in range(len(all_effects)):
    List_infra_actions_pre.append('List_%s_pre' %all_effects[i])
    List_infra_actions_pre[i] = ['List_%s_pre_%d' % (all_effects[i],index)]

# Lists needed for the infrastructure
List_instances = ['instances_0']
List_instance_eff = ['List_instance_eff_0']
List_instance_pre = ['List_instance_pre_0']
List_states = ['List_States_0']
List_actions = ['Action_0']
List_of_ml_inputs = ['ML_in_0']
List_of_ml_outputs = ['ML_out_0']

# Build array for initial and goal state
Init = z3.Array('Init', variables, z3.RealSort())
Goal = z3.Array('Goal', variables, z3.RealSort())

solver = z3.Solver()

# Define initial and goal state
for r in range(len(start_end_rows)):
    if ip.start_end.iloc[r,0] != "":
        inits = ip.start_end.iloc[r,0]
        solver.add(Init[list_of_var[r]] == eval(inits))
for r in range(len(start_end_rows)):
    if ip.start_end.iloc[r,1] != "":
        ends = ip.start_end.iloc[r,1]
        solver.add(Goal[list_of_var[r]] == eval(ends))

# Add parameter constraints
if 1 == 1:
    for k in range(len(para_fix_colums)):
        if para_fix_colums[k] == "Top_fix":
            for p in range(len(para_fix_rows)):
                solver.add(list_of_fix_para[p] <= float(ip.para_fix.iloc[p, k]))
        elif para_fix_colums[k] == "Down_fix":
            for p in range(len(para_fix_rows)):
                solver.add(list_of_fix_para[p] >= float(ip.para_fix.iloc[p, k]))
        else:
            for p in range(len(para_fix_rows)):
                solver.add(list_of_fix_para[p] == float(ip.para_fix.iloc[p, k]))

def neue_Instanz(index, state):
    # extend lists of variable parameters
    for p in range(len(list_of_var_para)):
        list_of_var_para[p].append('list_of_var_%s_%d' % (para_var_rows[p], index))
        list_of_var_para[p][index] = z3.Real('list_of_var_%s_%d' % (para_var_rows[p], index))

    # instanciate all parameters including their constraints
    for k in range(len(para_var_colums)):
        if para_var_colums[k] == "Top_var":
            for p in range(len(para_var_rows)):
                solver.add(list_of_var_para[p][index] <= float(ip.para_var.iloc[p, k]))
        elif para_var_colums[k] == "Down_var":
            for p in range(len(para_var_rows)):
                solver.add(list_of_var_para[p][index] >= float(ip.para_var.iloc[p, k]))
        else:
            for p in range(len(para_var_rows)):
                solver.add(list_of_var_para[p][index] == float(ip.para_var.iloc[p, k]))

    # new instance
    Number = z3.Datatype('current_instance_%d' % index)
    for i in range(len(List_number_actions)):
        List_number_actions[i] = Number.declare(all_effects[i])
    Number = Number.create()

    # Defintion of Arrays for the effects and preconditions
    for i in range(len(all_effects)):
        List_infra_actions_effects[i][index] = z3.Array('List_%s_eff_%d' % (all_effects[i],index), variables,
                                                        z3.RealSort())
        List_infra_actions_pre[i][index] = z3.Array('List_%s_pre_%d' % (all_effects[i], index), variables,
                                                        z3.RealSort())

    # Definition of the preconditions of symbolic actions
    for c in range(len(effect_colums)):
        for r in range(len(effect_rows)):
            if ip.precons.iloc[r,c] == "":
                break
            sep_pre = ip.precons.iloc[r, c].split('&')
            for s in sep_pre:
                if s[0] == '<':
                    if s[1] == '=':
                        ts = s[2:]
                        solver.add(List_infra_actions_pre[c][index][list_of_var[r]] <= eval(ts))
                    else:
                        ts = s[1:]
                        solver.add(List_infra_actions_pre[c][index][list_of_var[r]] < eval(ts))
                elif s[0] == '>':
                    if s[1] == '=':
                        ts = s[2:]
                        solver.add(List_infra_actions_pre[c][index][list_of_var[r]] >= eval(ts))
                    else:
                        ts = s[1:]
                        solver.add(List_infra_actions_pre[c][index][list_of_var[r]] > eval(ts))
                elif s[0] == '=':
                    print('Precon at Location %d and %d could not be considered' %(r,c))
                else:
                    solver.add(List_infra_actions_pre[c][index][list_of_var[r]] == eval(s))

    if integrate_learned_model == True:
        # Definition of the preconditions of learned actions
        for lc in range(len(ld_models_colums)):
            for lr in range(len(ld_models_rows)):
                if ip.L_precons.iloc[lr,lc] == "":
                    break
                sep_pre = ip.L_precons.iloc[lr,lc].split('&')
                for s in sep_pre:
                    if s[0] == '<':
                        if s[1] == '=':
                            ts = s[2:]
                            solver.add(List_infra_actions_pre[len(effect_colums)+lc][index][list_of_var[lr]] <= eval(ts))
                        else:
                            ts = s[1:]
                            solver.add(List_infra_actions_pre[len(effect_colums)+lc][index][list_of_var[lr]] < eval(ts))
                    elif s[0] == '>':
                        if s[1] == '=':
                            ts = s[2:]
                            solver.add(List_infra_actions_pre[len(effect_colums)+lc][index][list_of_var[lr]] >= eval(ts))
                        else:
                            ts = s[1:]
                            solver.add(List_infra_actions_pre[len(effect_colums)+lc][index][list_of_var[lr]] > eval(ts))
                    elif s[0] == '=':
                        print('Precon at Location %d and %d could not be considered' %(len(effect_rows)+lr,len(effect_colums)+lc))
                    else:
                        solver.add(List_infra_actions_pre[len(effect_colums)+lc][index][list_of_var[lr]] == eval(s))

    # Definition of the effects of symbolic actions
    for c in range(len(effect_colums)):
        for r in range(len(effect_rows)):
            if ip.effects.iloc[r, c] == "0":
                solver.add(List_infra_actions_effects[c][index][list_of_var[r]] == 0)
            else:
                sep_eff = ip.effects.iloc[r, c]
                for i in range(len(para_var_rows)):
                    y = '(list_of_var_para[' + str(i) + '][index])'
                    sep_eff = sep_eff.replace(para_var_rows[i], y)
                for i in range(len(para_fix_rows)):
                    x = '(list_of_fix_para[' + str(i) + '])'
                    sep_eff = sep_eff.replace(para_fix_rows[i], x)
                solver.add(List_infra_actions_effects[c][index][list_of_var[r]] == eval(sep_eff))

    if integrate_learned_model == True:
        # Definition of the effects of learned actions
        List_of_ml_inputs[index] = z3.Array('ML_in_%d' % index, ge.position, z3.RealSort())
        for i in range(len(list_of_var)):
            solver.add(state[list_of_var[i]] == List_of_ml_inputs[index][ge.list_of_positions[i]])
        for i in range(len(para_fix_rows)):
            solver.add(list_of_fix_para[i] == List_of_ml_inputs[index][ge.list_of_positions[len(list_of_var)+i]])
        for i in range(len(para_var_rows)):
            solver.add(list_of_var_para[i][index] == List_of_ml_inputs[index][ge.list_of_positions[len(list_of_var)+len(para_fix_rows)+i]])
        List_of_ml_outputs[index] = z3.Array('ML_out_%d' % index, ge.position, z3.RealSort())
        List_of_ml_outputs[index] = ge.nn_inf(index, solver, List_of_ml_inputs[index], ge.Weights)
        for i in range(len(list_of_var)):
            solver.add(List_infra_actions_effects[len(effect_colums)][index][list_of_var[i]] == List_of_ml_outputs[index][ge.list_of_positions[i]])

    # Arrays collecting the effects/preconditions of instances
    Instance_Effect = z3.Array('Instance_Eff', Number, z3.ArraySort(variables, z3.RealSort()))
    Instance_Precon = z3.Array('Instance_Pre', Number, z3.ArraySort(variables, z3.RealSort()))

    for c in range(len(all_effects)):
        Number_effekt = 'Number.' + all_effects[c]
        solver.add(Instance_Effect[eval(Number_effekt)] == List_infra_actions_effects[c][index])
        solver.add(Instance_Precon[eval(Number_effekt)] == List_infra_actions_pre[c][index])
    return Number

def extend_lists(index):
    List_instances.append("%d" % index)
    List_instance_eff.append('List_instance_eff_%d' % index)
    List_instance_pre.append('List_instance_pre_%d' % index)
    List_states.append('List_States_%d' % index)
    List_actions.append('Action_%d' % index)
    if integrate_learned_model == True:
        List_of_ml_inputs.append('ML_in_%d' % index)
        List_of_ml_outputs.append('ML_out_%d' % index)
    for i in range(len(all_effects)):
        List_infra_actions_effects[i].append('List_%s_eff_%d' % (all_effects[i], index))
        List_infra_actions_pre[i].append('List_%s_pre_%d' % (all_effects[i], index))

def print_all_variable_parameter(x):
    for i in range(len(List_instances)):
        for p in list_of_var_para:
            print( 'Parameter ', para_var_rows[list_of_var_para.index(p)], ' in instance ', i, ' : ', x[p[i]])

def print_schedule(x):
    out_schedule = pd.DataFrame(index=range(len(plan)), columns=range(len(para_var_rows)+1))
    for i in range(len(plan)):
        out_schedule[0][i] = x[plan[i]]
    for i in range(len(List_instances)):
        counter_val = 1
        for p in list_of_var_para:
            out_schedule[counter_val][i] = x[p[i]]
            counter_val = counter_val + 1
    out_schedule_columns = para_var_rows
    out_schedule_columns.insert(0, 'Action')
    out_schedule.index = plan
    out_schedule.columns = [out_schedule_columns]
    print('The plan and corresponding parameters:')
    print(out_schedule)

def const_to_dataframe(x):
    out_fix_para = pd.DataFrame(index=range(len(para_fix_rows)), columns=range(1))
    counter_val = 0
    for p in list_of_fix_para:
        out_fix_para[0][counter_val] = x[p]
        counter_val = counter_val + 1
    out_fix_para.index = para_fix_rows
    out_fix_para.columns = ['values']
    print(out_fix_para)

def print_const_parameter(x):
    for p in list_of_fix_para:
        print(p, 'is', x[p])

def solve(index):
    # main
    var = z3.Const('var', variables)
    List_instances[index] = neue_Instanz(index, Init)
    List_actions[index] = z3.Const('Action_%d' % index, List_instances[index])
    List_instance_pre[index] = z3.Array('Instance_Pre', List_instances[index], z3.ArraySort(variables, z3.RealSort()))
    List_instance_eff[index] = z3.Array('Instance_Eff', List_instances[index], z3.ArraySort(variables, z3.RealSort()))
    List_states[index] = z3.Array('States', List_instances[index], z3.ArraySort(variables, z3.RealSort()))
    plan.append(List_actions[index])
    solver.add(z3.ForAll([var], List_instance_pre[index][List_actions[index]][var] == Init[var]))
    solver.add(z3.ForAll([var], List_states[index][List_actions[index]][var] == Init[var] +
                         List_instance_eff[index][List_actions[index]][var]))
    while index < max_step_number:
        solver.push()
        if ip.approx == True:
            #this is necessary, if small ML models are integrated, due to their inaccuracy 
            solver.add(z3.ForAll([var], z3.And(List_states[index][List_actions[index]][var] <= Goal[var]+2, List_states[index][List_actions[index]][var] >= Goal[var]-2)))
        else:
            solver.add(z3.ForAll([var], List_states[index][List_actions[index]][var] == Goal[var]))
        if z3.sat == solver.check():
            print('Sat')
            m = solver.model()
            print_all_variable_parameter(m)
            const_to_dataframe(m)
            print_schedule(m)
            break
        solver.pop()
        index = index + 1
        end_ex = datetime.now()
        print('There was no solution found in instance %d.' % index)
        print(end_ex - start_ex)
        extend_lists(index)
        List_instances[index] = neue_Instanz(index, List_states[index-1][List_actions[index-1]])
        List_actions[index] = z3.Const('Action_%d' % index, List_instances[index])
        List_instance_pre[index] = z3.Array('Instance_Pre', List_instances[index], z3.ArraySort(variables, z3.RealSort()))
        List_instance_eff[index] = z3.Array('Instance_Eff', List_instances[index], z3.ArraySort(variables, z3.RealSort()))
        List_states[index] = z3.Array('States', List_instances[index], z3.ArraySort(variables, z3.RealSort()))
        plan.append(List_actions[index])
        solver.add(z3.ForAll([var], List_instance_pre[index][List_actions[index]][var] ==
                             List_states[index - 1][List_actions[index - 1]][var]))
        solver.add(z3.ForAll([var], List_states[index][List_actions[index]][var] ==
                             List_states[index - 1][List_actions[index - 1]][var] +
                             List_instance_eff[index][List_actions[index]][var]))
    if index >= max_step_number:
        print('There is no solution within the maximum number of steps.')

solve(index)