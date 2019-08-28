from keras.layers import *
from keras.models import *


def attention_block(inputs_atten, channels_num2, time_step2):
    a = inputs_atten
    a = Dense(channels_num2, activation='sigmoid')(a)
    a = Lambda(lambda x: K.mean(x, axis=1))(a)
    a = RepeatVector(time_step2)(a)
    output_attention_mul = multiply([inputs_atten, a])
    return output_attention_mul


def renet_block(x, kernel, deepth=4, pool='MAX'):
    x = Conv1D(filters=16, kernel_size=1, strides=1,  activation='linear', padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)

    x_short = x
    x = Conv1D(filters=16, kernel_size=kernel, strides=1, activation='linear', padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    # x = BatchNormalization()(x)
    x = Conv1D(filters=16, kernel_size=kernel, strides=1, activation='linear', padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)

    if pool == 'MAX':
        x = MaxPool1D(pool_size=2)(x)
    else:
        x = AveragePooling1D(pool_size=2)(x)

    x_short = Conv1D(filters=16, kernel_size=1, strides=1, activation='linear', padding='same')(x_short)
    if pool == 'MAX':
        x_short = MaxPool1D(pool_size=2)(x_short)
    else:
        x_short = AveragePooling1D(pool_size=2)(x_short)

    x = add([x, x_short])
    x = Conv1D(filters=64, kernel_size=1, strides=1, activation='linear', padding='same')(x)
    for i in range(deepth):
        x_short = x
        x = Conv1D(filters=64, kernel_size=kernel, strides=1, activation='linear', padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv1D(filters=64, kernel_size=kernel, strides=1, activation='linear', padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)

        if pool == 'MAX':
            x = MaxPool1D(pool_size=2)(x)
        else:
            x = AveragePooling1D(pool_size=2)(x)

        x_short = Conv1D(filters=64, kernel_size=1, strides=1, activation='linear', padding='same')(x_short)
        if pool == 'MAX':
            x_short = MaxPool1D(pool_size=2)(x_short)
        else:
            x_short = AveragePooling1D(pool_size=2)(x_short)

        x = add([x, x_short])
    return x


# 片段特征提取模型
def CNN_model(channels_num, time_step, output, is_training=0, poolway='MAX'):
    X_inputs = (time_step, channels_num)
    X_inputs = Input(X_inputs)
    deepth = 5
    tmp = 1
    x = X_inputs

    x1 = renet_block(x, 2, deepth, poolway)
    x2 = renet_block(x, 4, deepth, poolway)
    x5 = renet_block(x, 8, deepth, poolway)
    x3 = renet_block(x, 16, deepth, poolway)
    x4 = renet_block(x, 32, deepth)

    x1 = attention_block(x1, 64, int(78/tmp))
    x2 = attention_block(x2, 64, int(78/tmp))
    x3 = attention_block(x3, 64, int(78/tmp))
    x4 = attention_block(x4, 64, int(78/tmp))
    x5 = attention_block(x5, 64, int(78/tmp))
    x = Concatenate(axis=2)([x1, x2, x3, x4, x5])
    x = TimeDistributed(Dense(256))(x)
    x = GlobalAveragePooling1D()(x)

    x = Dropout(0.5)(x)

    x = Dense(output, activation='sigmoid')(x)
    model = Model(inputs=X_inputs, outputs=x, name='atten_cnn')
    return model


# 片段特征整合模型
def myLSTM(feature_num, time_step, mask_num):
    X_input = Input((time_step, feature_num))
    x = Masking(mask_value=mask_num, input_shape=(time_step, feature_num))(X_input)
    # x = attention_block3(x, time_step, 256)
    # x = TimeDistributed(Flatten())(x)
    x1 = Bidirectional(LSTM(200, return_sequences=True))(x)
    x1 = Bidirectional(LSTM(200, return_sequences=True))(x1)
    x1 = Bidirectional(LSTM(200, return_sequences=True))(x1)
    # x1 = attention_block3(x1, time_step, 400)
    x1 = TimeDistributed(Dense(400))(x1)
    x1 = GlobalAveragePooling1D()(x1)

    x = x1

    x = Dropout(0.5)(x)
    x = Dense(10, activation='sigmoid')(x)
    model = Model(inputs=X_input, outputs=x, name='atten_cnn')
    return model
