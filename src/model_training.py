# model_training.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, DepthwiseConv2D, Activation, AveragePooling2D, Dropout, SeparableConv2D, Flatten, Dense
from tensorflow.keras.constraints import max_norm
from scikeras.wrappers import KerasClassifier
import optuna
from sklearn.model_selection import GridSearchCV

def create_eegnet_model(dropoutRate, kernLength, F1, D, F2, input_shape, nb_classes):
    """
    Build and compile an EEGNet model with given hyperparameters.
    """
    input1 = Input(shape=input_shape)
    
    # First block: Temporal convolution followed by depthwise convolution.
    x = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((input_shape[0], 1), use_bias=False,
                        depth_multiplier=D,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropoutRate)(x)
    
    # Second block: Separable convolution.
    x = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropoutRate)(x)
    
    # Classification block.
    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(0.5))(x)
    output = Activation('softmax')(x)
    
    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_model_for_scikeras(dropoutRate, kernLength, F1, D, F2, input_shape, nb_classes):
    """
    Wrapper function for scikeras's KerasClassifier.
    """
    return create_eegnet_model(dropoutRate, kernLength, F1, D, F2, input_shape, nb_classes)

def objective(trial, X_train, y_train, X_val, y_val, input_shape, nb_classes, epochs=20, batch_size=32):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    # Define a broad search space.
    dropoutRate = trial.suggest_float('dropoutRate', 0.2, 0.5)
    kernLength = trial.suggest_int('kernLength', 16, 64, step=8)
    F1 = trial.suggest_int('F1', 4, 16, step=2)
    D = trial.suggest_int('D', 1, 2)
    F2 = trial.suggest_int('F2', 8, 32, step=4)
    
    # Build and train the model using the suggested hyperparameters.
    model = create_eegnet_model(dropoutRate, kernLength, F1, D, F2, input_shape, nb_classes)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Use the last validation accuracy as the metric (Optuna minimizes the objective, so return 1 - accuracy).
    val_accuracy = history.history['val_accuracy'][-1]
    return 1.0 - val_accuracy

def optuna_hyperparameter_search(X_train, y_train, X_val, y_val, input_shape, nb_classes, n_trials=20):
    """
    Run an Optuna study to explore a broad range of hyperparameters.
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_shape, nb_classes), n_trials=n_trials)
    
    best_params = study.best_trial.params
    print("Best Optuna Params:", best_params)
    return best_params

def grid_search_finetune(X_train, y_train, input_shape, nb_classes, epochs=50, batch_size=32, cv=3, refined_grid=None):
    """
    Run GridSearchCV to fine-tune hyperparameters.
    Optionally, a refined_grid dictionary can be provided.
    """
    # If no refined grid is provided, define a default grid.
    if refined_grid is None:
        refined_grid = {
            'dropoutRate': [0.25, 0.3, 0.35],
            'kernLength': [24, 32, 40],
            'F1': [6, 8, 10],
            'D': [1, 2],
            'F2': [12, 16, 20],
            'epochs': [epochs],
            'batch_size': [batch_size],
        }
    
    # Wrap the model building function using scikeras.
    model_clf = KerasClassifier(
        model=build_model_for_scikeras,
        input_shape=input_shape,
        nb_classes=nb_classes,
        verbose=0,
    )
    
    grid = GridSearchCV(estimator=model_clf, param_grid=refined_grid, cv=cv, n_jobs=1)
    grid_result = grid.fit(X_train, y_train)
    
    print("Best GridSearch Params:", grid_result.best_params_)
    print("Best Score:", grid_result.best_score_)
    
    return grid_result

def train_model_with_tuning(X_train, y_train, X_val, y_val, input_shape, nb_classes):
    """
    Use a two-step approach:
      1. Use Optuna to determine a good hyperparameter range.
      2. Fine-tune using GridSearchCV within that range.
    """
    # Step 1: Optuna search.
    best_optuna_params = optuna_hyperparameter_search(X_train, y_train, X_val, y_val, input_shape, nb_classes, n_trials=20)
    
    # Build a refined grid around the best parameters.
    dropout_best = best_optuna_params['dropoutRate']
    kernLength_best = best_optuna_params['kernLength']
    F1_best = best_optuna_params['F1']
    D_best = best_optuna_params['D']
    F2_best = best_optuna_params['F2']
    
    refined_grid = {
        'dropoutRate': [max(0.2, dropout_best - 0.05), dropout_best, min(0.5, dropout_best + 0.05)],
        'kernLength': [max(16, kernLength_best - 8), kernLength_best, kernLength_best + 8],
        'F1': [max(4, F1_best - 2), F1_best, F1_best + 2],
        'D': [D_best] if D_best == 2 else [D_best, D_best + 1],
        'F2': [max(8, F2_best - 4), F2_best, F2_best + 4],
        'epochs': [50],
        'batch_size': [32],
    }
    
    # Step 2: GridSearch fine tuning.
    grid_result = grid_search_finetune(X_train, y_train, input_shape, nb_classes, epochs=50, batch_size=32, cv=3, refined_grid=refined_grid)
    
    return grid_result
