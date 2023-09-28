import pandas as pd
import torch
import ML_train_FliPSi as ml
import z3

# This file is the implementation of the model in SMT.
# The weights of the trained model are extracted and feed into this representation. 

def nn_parameters_to_df(nn_model):
    df_list = []
    for name, param in nn_model.named_parameters():
        param_data = {
            "Layer": name,
            "Shape": param.size(),
            "Values": param.detach().numpy()
        }
        df_list.append(param_data)
    df = pd.DataFrame(df_list)
    return df

PATH = "model_FliPSi.pth"

model = ml.MLP(ml.hparam)
model.load_state_dict(torch.load(PATH))
model.eval()

def predict(model, x):
    model.eval()
    with torch.no_grad():
        P_out = model(x)
    return P_out

def test_model(model, features, p_fix, p_var): 
    P_in = torch.cat((features, p_fix, p_var), dim=0)
    print('Input')
    print(P_in)
    P_out = predict(model, P_in)
    print('Inference')
    print(P_out)

# extracting the weights of the model
Weights = nn_parameters_to_df(model)

positions = ['posi_1','posi_2','posi_3','posi_4','posi_5','posi_6','posi_7','posi_8','posi_9']
list_of_positions = []
position, list_of_positions = z3.EnumSort('position', positions)

prepos = ['prepo_1','prepo_2','prepo_3','prepo_4','prepo_5','prepo_6','prepo_7','prepo_8','prepo_9']
list_of_prepos = []
prepo, list_of_prepos = z3.EnumSort('prepo', prepos)

List_P_L1 = []
List_P_L2 = []
List_P_L4 = []

List_W_L1 = []
List_W_L2 = []
List_W_L4 = []

List_B_L1 = []
List_B_L2 = []
List_B_L4 = []

def nn_inf(call, s, P_in, Weights):
    n_N_L_1 = 9
    n_N_L_2 = 9
    n_N_L_4 = 5

    List_P_L1.append('P_L1_%d' % call)
    List_P_L2.append('P_L2_%d' % call)
    List_P_L4.append('P_L4_%d' % call)

    List_W_L1.append('W_L1_%d' % call)
    List_W_L2.append('W_L2_%d' % call)
    List_W_L4.append('W_L4_%d' % call)

    List_B_L1.append('B_L1_%d' % call)
    List_B_L2.append('B_L2_%d' % call)
    List_B_L4.append('B_L4_%d' % call)


    List_P_L1[call] = z3.Array('P_L1_%d' % call, position, z3.RealSort())
    List_P_L2[call] = z3.Array('P_L2_%d' % call, position, z3.RealSort())
    List_P_L4[call] = z3.Array('P_L4_%d' % call, position, z3.RealSort())

    List_W_L1[call] = z3.Array('W_L1_%d' % call, position, z3.ArraySort(prepo, z3.RealSort()))
    List_W_L2[call] = z3.Array('W_L2_%d' % call, position, z3.ArraySort(prepo, z3.RealSort()))
    List_W_L4[call] = z3.Array('W_L4_%d' % call, position, z3.ArraySort(prepo, z3.RealSort()))

    ls_W_L1 = []
    for i in range(n_N_L_1): 
        ls_W_L1.append('ls_W_L1_%d' %i)
        ls_W_L1[i] = z3.Array('ls_W_L1_%d' %i, prepo, z3.RealSort())

    ls_W_L2 = []
    for i in range(n_N_L_2): 
        ls_W_L2.append('ls_W_L2_%d' %i)
        ls_W_L2[i] = z3.Array('ls_W_L2_%d' %i, prepo, z3.RealSort())

    ls_W_L4 = []
    for i in range(n_N_L_4): 
        ls_W_L4.append('ls_W_L4_%d' %i)
        ls_W_L4[i] = z3.Array('ls_W_L4_%d' %i, prepo, z3.RealSort())

    List_B_L1[call] = z3.Array('B_L1_%d' % call, position, z3.ArraySort(prepo, z3.RealSort()))
    List_B_L2[call] = z3.Array('B_L2_%d' % call, position, z3.ArraySort(prepo, z3.RealSort()))
    List_B_L4[call] = z3.Array('B_L4_%d' % call, position, z3.ArraySort(prepo, z3.RealSort()))

    ls_B_L1 = []
    for i in range(n_N_L_1): 
        ls_B_L1.append('ls_B_L1_%d' %i)
        ls_B_L1[i] = z3.Array('ls_B_L1_%d' %i, prepo, z3.RealSort())

    ls_B_L2 = []
    for i in range(n_N_L_2): 
        ls_B_L2.append('ls_B_L2_%d' %i)
        ls_B_L2[i] = z3.Array('ls_B_L2_%d' %i, prepo, z3.RealSort())

    ls_B_L4 = []
    for i in range(n_N_L_4): 
        ls_B_L4.append('ls_B_L4_%d' %i)
        ls_B_L4[i] = z3.Array('ls_B_L4_%d' %i, prepo, z3.RealSort())

    for i in range(n_N_L_1):
        s.add(List_W_L1[call][list_of_positions[i]] == ls_W_L1[i])
        s.add(List_B_L1[call][list_of_positions[i]] == ls_B_L1[i])

    for i in range(n_N_L_2):
        s.add(List_W_L2[call][list_of_positions[i]] == ls_W_L2[i])
        s.add(List_B_L2[call][list_of_positions[i]] == ls_B_L2[i])

    for i in range(n_N_L_4):
        s.add(List_W_L4[call][list_of_positions[i]] == ls_W_L4[i])
        s.add(List_B_L4[call][list_of_positions[i]] == ls_B_L4[i])

    for p in range(len(positions)):
        for pp in range(len(prepos)):
            s.add(List_W_L1[call][list_of_positions[p]][list_of_prepos[pp]] == Weights.iloc[0][2][p][pp])   # Weights of the first layer
            s.add(List_W_L2[call][list_of_positions[p]][list_of_prepos[pp]] == Weights.iloc[2][2][p][pp])   # Weights of the second layer
            s.add(List_B_L1[call][list_of_positions[p]][list_of_prepos[pp]] == Weights.iloc[1][2][p])       # Biases of the first layer
            s.add(List_B_L2[call][list_of_positions[p]][list_of_prepos[pp]] == Weights.iloc[3][2][p])       # Biases of the second layer

    for p in range(n_N_L_4):
        for pp in range(len(prepos)):
            s.add(List_W_L4[call][list_of_positions[p]][list_of_prepos[pp]] == Weights.iloc[6][2][p][pp])   # Weights of the 4th layer
            s.add(List_B_L4[call][list_of_positions[p]][list_of_prepos[pp]] == Weights.iloc[7][2][p])       # Biases of the 4th layer

    for p in range(n_N_L_1):
        s.add(z3.Implies((P_in[list_of_positions[0]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[0]]
                        + P_in[list_of_positions[1]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[1]]
                        + P_in[list_of_positions[2]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[2]]
                        + P_in[list_of_positions[3]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[3]]
                        + P_in[list_of_positions[4]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[4]]
                        + P_in[list_of_positions[5]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[5]]
                        + P_in[list_of_positions[6]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[6]]
                        + P_in[list_of_positions[7]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[7]]
                        + P_in[list_of_positions[8]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[8]]
                        + List_B_L1[call][list_of_positions[p]][list_of_prepos[p]]
                        ) <= 0, List_P_L1[call][list_of_positions[p]] == 0))
        s.add(z3.Implies((P_in[list_of_positions[0]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[0]]
                        + P_in[list_of_positions[1]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[1]]
                        + P_in[list_of_positions[2]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[2]]
                        + P_in[list_of_positions[3]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[3]]
                        + P_in[list_of_positions[4]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[4]]
                        + P_in[list_of_positions[5]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[5]]
                        + P_in[list_of_positions[6]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[6]]
                        + P_in[list_of_positions[7]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[7]]
                        + P_in[list_of_positions[8]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[8]]
                        + List_B_L1[call][list_of_positions[p]][list_of_prepos[p]]
                        ) > 0, List_P_L1[call][list_of_positions[p]] == 
                          P_in[list_of_positions[0]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[0]]
                        + P_in[list_of_positions[1]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[1]]
                        + P_in[list_of_positions[2]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[2]]
                        + P_in[list_of_positions[3]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[3]]
                        + P_in[list_of_positions[4]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[4]]
                        + P_in[list_of_positions[5]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[5]]
                        + P_in[list_of_positions[6]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[6]]
                        + P_in[list_of_positions[7]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[7]]
                        + P_in[list_of_positions[8]] * List_W_L1[call][list_of_positions[p]][list_of_prepos[8]]
                        + List_B_L1[call][list_of_positions[p]][list_of_prepos[p]]
                        ))
        
    for p in range(n_N_L_2):
        s.add(z3.Implies((List_P_L1[call][list_of_positions[0]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[0]]
                        + List_P_L1[call][list_of_positions[1]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[1]]
                        + List_P_L1[call][list_of_positions[2]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[2]]
                        + List_P_L1[call][list_of_positions[3]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[3]]
                        + List_P_L1[call][list_of_positions[4]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[4]]
                        + List_P_L1[call][list_of_positions[5]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[5]]
                        + List_P_L1[call][list_of_positions[6]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[6]]
                        + List_P_L1[call][list_of_positions[7]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[7]]
                        + List_P_L1[call][list_of_positions[8]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[8]] 
                        + List_B_L2[call][list_of_positions[p]][list_of_prepos[p]]
                        ) <= 0, List_P_L2[call][list_of_positions[p]] == 0))
        s.add(z3.Implies((List_P_L1[call][list_of_positions[0]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[0]]
                        + List_P_L1[call][list_of_positions[1]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[1]]
                        + List_P_L1[call][list_of_positions[2]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[2]]
                        + List_P_L1[call][list_of_positions[3]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[3]]
                        + List_P_L1[call][list_of_positions[4]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[4]]
                        + List_P_L1[call][list_of_positions[5]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[5]]
                        + List_P_L1[call][list_of_positions[6]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[6]]
                        + List_P_L1[call][list_of_positions[7]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[7]]
                        + List_P_L1[call][list_of_positions[8]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[8]] 
                        + List_B_L2[call][list_of_positions[p]][list_of_prepos[p]]
                        ) > 0, List_P_L2[call][list_of_positions[p]] == 
                          List_P_L1[call][list_of_positions[0]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[0]]
                        + List_P_L1[call][list_of_positions[1]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[1]]
                        + List_P_L1[call][list_of_positions[2]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[2]]
                        + List_P_L1[call][list_of_positions[3]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[3]]
                        + List_P_L1[call][list_of_positions[4]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[4]]
                        + List_P_L1[call][list_of_positions[5]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[5]]
                        + List_P_L1[call][list_of_positions[6]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[6]]
                        + List_P_L1[call][list_of_positions[7]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[7]]
                        + List_P_L1[call][list_of_positions[8]] * List_W_L2[call][list_of_positions[p]][list_of_prepos[8]]
                        + List_B_L2[call][list_of_positions[p]][list_of_prepos[p]]
                        ))
        
    for p in range(n_N_L_4):
        s.add(z3.Implies((List_P_L2[call][list_of_positions[0]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[0]]
                        + List_P_L2[call][list_of_positions[1]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[1]]
                        + List_P_L2[call][list_of_positions[2]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[2]]
                        + List_P_L2[call][list_of_positions[3]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[3]]
                        + List_P_L2[call][list_of_positions[4]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[4]]
                        + List_P_L2[call][list_of_positions[5]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[5]]
                        + List_P_L2[call][list_of_positions[6]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[6]]
                        + List_P_L2[call][list_of_positions[7]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[7]]
                        + List_P_L2[call][list_of_positions[8]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[8]]
                        + List_B_L4[call][list_of_positions[p]][list_of_prepos[p]]
                        ) <= 2, List_P_L4[call][list_of_positions[p]] == 0))
        s.add(z3.Implies((List_P_L2[call][list_of_positions[0]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[0]]
                        + List_P_L2[call][list_of_positions[1]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[1]]
                        + List_P_L2[call][list_of_positions[2]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[2]]
                        + List_P_L2[call][list_of_positions[3]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[3]]
                        + List_P_L2[call][list_of_positions[4]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[4]]
                        + List_P_L2[call][list_of_positions[5]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[5]]
                        + List_P_L2[call][list_of_positions[6]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[6]]
                        + List_P_L2[call][list_of_positions[7]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[7]]
                        + List_P_L2[call][list_of_positions[8]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[8]]
                        + List_B_L4[call][list_of_positions[p]][list_of_prepos[p]]
                        ) > 2, List_P_L4[call][list_of_positions[p]] == 
                          List_P_L2[call][list_of_positions[0]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[0]]
                        + List_P_L2[call][list_of_positions[1]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[1]]
                        + List_P_L2[call][list_of_positions[2]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[2]]
                        + List_P_L2[call][list_of_positions[3]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[3]]
                        + List_P_L2[call][list_of_positions[4]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[4]]
                        + List_P_L2[call][list_of_positions[5]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[5]]
                        + List_P_L2[call][list_of_positions[6]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[6]]
                        + List_P_L2[call][list_of_positions[7]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[7]]
                        + List_P_L2[call][list_of_positions[8]] * List_W_L4[call][list_of_positions[p]][list_of_prepos[8]]
                        + List_B_L4[call][list_of_positions[p]][list_of_prepos[p]]
                        ))
        
    return List_P_L4[call]