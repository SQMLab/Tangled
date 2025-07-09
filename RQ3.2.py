import numpy as np
import pandas as pd
from tqdm import tqdm
import ast

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
import gc
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model, Input
import numpy as np
import tensorflow.keras.backend as K

random_seed = 42

import warnings
warnings.filterwarnings('ignore')

def get_model():
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    tf.config.run_functions_eagerly(False)

    input_shape = (768,)

    # Commit Message Classification Model
    input_a = Input(shape=input_shape, name="cls_input")
    x_a = layers.Dense(512, activation='relu', name="cls_l1")(input_a)
    x_a = layers.Dense(256, activation='relu', name="cls_l2")(x_a)
    x_a = layers.Dense(64, activation='relu', name="cls_l3")(x_a)
    x_a = layers.Dense(32, activation='relu', name="cls_l4")(x_a)
    class_output = layers.Dense(1, activation='sigmoid', name="cls_output")(x_a)


    # Commit Message and Diff Similarity Model
    input_b = Input(shape=input_shape, name="sim_input_1")
    input_c = Input(shape=input_shape, name="sim_input_2")

    def shared_siamese_network(input_layer, name):
        x = layers.Dense(512, activation='relu', name=f"sim_{name}_1")(input_layer)
        x = layers.Dense(256, activation='relu', name=f"sim_{name}_2")(x)
        x = layers.Dense(64, activation='linear', name=f"sim_{name}_3")(x) 
        output = layers.Dense(32, activation='linear', name=f"sim_{name}_4")(x) 
        return output

    embedding_b = shared_siamese_network(input_b, "1")
    embedding_c = shared_siamese_network(input_c, "2")

    def similarity(vectors):
        x, y = vectors
        x = tf.nn.l2_normalize(x, axis=1) 
        y = tf.nn.l2_normalize(y, axis=1)
        return tf.reduce_sum(x * y, axis=1, keepdims=True)
    
    # Y = A_.B_ + A
    def logical_op(vectors):
        a, b = vectors
        #b = b * b
        a = 1 - a
        b = 1 - b
        c = layers.Multiply(name="multiply")([a, b])
        c = layers.Add(name="addition")([c , vectors[0]])
        return c
    
    similarity_output = layers.Lambda(similarity, name="similarity_output")([embedding_b, embedding_c])
    #similarity_output = layers.Dense(1, activation="sigmoid", name="similarity_output")(similarity_output)

    similarity_vec = layers.Subtract(name = "subtract")([embedding_b, embedding_c])
    similarity_vec = layers.Dense(32, activation='relu', name="similarity_dense")(similarity_vec)

    final_output = layers.Concatenate(name="concatenate")([similarity_vec, x_a])
    final_output = layers.Dense(16, activation='relu', name="final_dense1")(final_output)
    final_output = layers.Dense(8, activation='relu', name="final_dense2")(final_output)
    final_output = layers.Dense(1, activation='sigmoid', name="final_output")(final_output)

    model = Model(inputs=[input_a, input_b, input_c], outputs=[class_output, similarity_output, final_output], name="model")

    cls_layers = [model.get_layer("cls_input"), model.get_layer("cls_l1"), model.get_layer("cls_l2"), model.get_layer("cls_l3"), model.get_layer("cls_l4"), model.get_layer("cls_output")]
    sim_layers = [model.get_layer("sim_input_1"), model.get_layer("sim_input_2"), 
                model.get_layer("sim_1_1"), model.get_layer("sim_1_2"), model.get_layer("sim_1_3"), model.get_layer("sim_1_4"),
                model.get_layer("sim_2_1"), model.get_layer("sim_2_2"), model.get_layer("sim_2_3"), model.get_layer("sim_1_4"),
                model.get_layer("similarity_output")
                ]
    final_layers = [model.get_layer("subtract"), model.get_layer("similarity_dense"), model.get_layer("concatenate"), model.get_layer("final_dense1"), model.get_layer("final_dense2"), model.get_layer("final_output")]

    optimizers_and_layers = [(tf.keras.optimizers.Adam(learning_rate=0.001), cls_layers),
                            (tf.keras.optimizers.SGD(learning_rate=0.01), sim_layers),
                            (tf.keras.optimizers.Adam(learning_rate=0.001), final_layers)]

    # Losses
    losses = {
        "cls_output":  tf.keras.losses.BinaryCrossentropy(from_logits=False),
        "similarity_output":  "mse",
        "final_output":  tf.keras.losses.BinaryCrossentropy(from_logits=False)  
    }

    # Metrics
    metrics = {
        "cls_output": "accuracy",
        "similarity_output": "mae",
        "final_output": "accuracy"
    }

    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics
    )

    return model


def prepare_data(msg_cls, msg_sim, diff_sim, cls_label, label, similarity):
    input_data_a = np.array([np.array(lst) for lst in msg_cls.values])
    input_data_b = np.array([np.array(lst) for lst in msg_sim.values])
    input_data_c = np.array([np.array(lst) for lst in diff_sim.values])

    class_labels = cls_label.values
    similarity_labels = similarity.values
    final_labels = [1 if x=="NotBuggy" else 0 for x in label.values]

    input_data_a = np.array(input_data_a, dtype=np.float32)
    input_data_b = np.array(input_data_b, dtype=np.float32)
    input_data_c = np.array(input_data_c, dtype=np.float32)

    class_labels = np.array(class_labels, dtype=np.float32).reshape(-1,1)
    similarity_labels = np.array(similarity_labels, dtype=np.float32).reshape(-1,1)
    final_labels = np.array(final_labels, dtype=np.float32).reshape(-1,1)

    return input_data_a, input_data_b, input_data_c, class_labels, similarity_labels, final_labels

df_emb = pd.read_csv("./data/Embeddings/MsgCls_MsgDiffSim.csv")
df_emb["Msg_Cls_Embeddings"] = df_emb["Msg_Cls_Embeddings"].apply(ast.literal_eval)
df_emb["Msg_Sim_Embeddings"] = df_emb["Msg_Sim_Embeddings"].apply(ast.literal_eval)
df_emb["Diff_Sim_Embeddings"] = df_emb["Diff_Sim_Embeddings"].apply(ast.literal_eval)

g = pd.read_csv("./data/GoldSet.csv")
tangled_buggy = df_emb[df_emb['Hash'].isin(g['CommitHash'])]
tangled_buggy["Similarity"] = tangled_buggy["Label"].apply(lambda x: 0 if x=="NotBuggy" else 1)
tangled_buggy["Cls_label"] = 0

true_buggy_notbuggy = df_emb[~df_emb['Hash'].isin(g['CommitHash'])]
true_buggy_notbuggy["Similarity"] = 1
true_buggy_notbuggy["Cls_label"] = true_buggy_notbuggy["Label"].apply(lambda x: 1 if x=="NotBuggy" else 0)

df_emb = pd.concat([tangled_buggy, true_buggy_notbuggy])

del g, tangled_buggy, true_buggy_notbuggy

y_preds, y_trues = [], []

vnt = 0
cv = LeaveOneOut()
for train_indx, test_indx in tqdm(cv.split(df_emb), total=df_emb.shape[0]):
    train = df_emb.iloc[train_indx]
    test = df_emb.iloc[test_indx]
    
    train_x_msg_cls, train_x_msg_sim, train_x_diff_sim, train_y_cls, train_y_sim, train_y = prepare_data(train["Msg_Cls_Embeddings"], train["Msg_Sim_Embeddings"], train["Diff_Sim_Embeddings"], train["Cls_label"], train["Label"], train["Similarity"])
    test_x_msg_cls, test_x_msg_sim, test_x_diff_sim, test_y_cls, test_y_sim, test_y = prepare_data(test["Msg_Cls_Embeddings"], test["Msg_Sim_Embeddings"], test["Diff_Sim_Embeddings"], test["Cls_label"], test["Label"], test["Similarity"]) 

    del train, test
    gc.collect()

    model = get_model()
    model.fit([train_x_msg_cls, train_x_msg_sim, train_x_diff_sim], 
                        {
                            "cls_output": train_y_cls, 
                            "similarity_output": train_y_sim, 
                            "final_output": train_y
                        }, 
                        epochs=15, 
                        batch_size=8,
                        verbose = False
                    )
    output_cls, output_sim, output = model.predict([test_x_msg_cls, test_x_msg_sim, test_x_diff_sim], verbose=False)
    y_pred = int(output[0][0] >= 0.5)

    y_preds.append(y_pred)
    y_trues.append(int(test_y[0]))

    del model, output_cls, output_sim, output
    del train_x_msg_cls, train_x_msg_sim, train_x_diff_sim, train_y_cls, train_y_sim, train_y
    del test_x_msg_cls, test_x_msg_sim, test_x_diff_sim, test_y_cls, test_y_sim, test_y
    K.clear_session()
    gc.collect()

    pd.DataFrame({"y_trues": y_trues, "y_preds":y_preds}).to_csv("./Results/RQ3/MultiNetwork_LOO_Result.csv", index = False)
    # vnt+=1
    # if vnt >2:
    #     break
    
print(
    "Accuracy", accuracy_score(y_trues, y_preds),
    "\nPrecision", precision_score(y_trues, y_preds, pos_label=0),
    "\nRecall", recall_score(y_trues, y_preds, pos_label=0),
    "\nF1 Score", f1_score(y_trues, y_preds, pos_label=0),
    "\nMCC", matthews_corrcoef(y_trues, y_preds),
)

print("\nClassification Report:\n", classification_report(y_trues, y_preds))
conf_matrix = confusion_matrix(y_trues, y_preds)
ConfusionMatrixDisplay(conf_matrix, display_labels=["Buggy", "NotBuggy"]).plot()
