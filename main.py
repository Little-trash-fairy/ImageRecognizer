import plaidml.keras

plaidml.keras.install_backend()

import matplotlib.pyplot as plt
from keras.datasets import cifar10  # Набор данных cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# Устанавливаем бэкенд для работы на AMD

# Количество классов изображений
Number_of_element_classes = 10

# Загрузка базы данных
# cifar10 - это база данных, содержащая в себе
# около 50-60 тысяч изображений низкого разрешения,
# которые можно можно разделить на 10 классов:
# 1. Самолёт
# 2. Автомобиль
# 3. Птица
# 4. Кошка
# 5. Олень
# 6. Собака
# 7. Жаба
# 8. Лошадь
# 9. Корабль
# 10. Грузовик
# Как уже было сказано ранее, изображения очень маленькие
# всего 32 * 32, зато представлены в цвете(RGB)
(training_in, trainin_out), (test_in, test_out) = cifar10.load_data()

# Приводим данные типа int в тип float32
# И нормализуем их, деление на 255, так как
# данными являются пиксели в диапазоне от 0 до 255
training_in = training_in.astype('float32')
test_in = test_in.astype('float32')
training_in /= 255
test_in /= 255

# Производим унитарное кодирование данных.
# Можем себе это позволить, так как в данной
# Выборке данных нет картинок, который могут занимать
# Промежуточное между классами состояние.
trainin_out = np_utils.to_categorical(trainin_out, Number_of_element_classes)
test_out = np_utils.to_categorical(test_out, Number_of_element_classes)


def model_init():
    # Создание модели сети
    model = Sequential()

    # Реализуем первый свёрточный слой
    model.add(Conv2D(
        32,  # Количество каналов
        (3, 3),  # Размер канала
        input_shape=training_in.shape[1:],  # Форма входа
        activation='relu',  # Устанавливаем функция активации, где F(x) = max(0,x)
        padding='same'))  # Не меняем размер изображения
    model.add(Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Реализуем исключающий слой, необходимый для
    # предотвращения появления переобучения
    # Принцип работы: случайным образом устраняет 25% соединений
    model.add(Dropout(0.25))

    model.add(Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        128,
        (3, 3),
        activation='relu',
        padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(
        128,
        (3, 3),
        activation='relu',
        padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(Number_of_element_classes, activation='softmax'))
    return model


if __name__ == "__main__":
    model = model_init()

    model.compile(loss='categorical_crossentropy',  # Функция потерь
                  optimizer='adam',  # Оптимизатор обучения
                  metrics=['accuracy'])  # Функция, используемая для оценки эффективности сети

    history = model.fit(training_in, trainin_out,
                        epochs=50,  # Количество полных прогонок данных
                        batch_size=128,  # Размер пакета обучения
                        validation_data=(test_in, test_out),  # Данные
                        shuffle=128,  # Либо True, тогда данные при каждой эпохе перемешиваются
                        # Либо число, равное batch_size за раз, тогда перемешивается только он
                        verbose=2,  # Тип вывода шагов прогонки данных
                        validation_split=0.1)  # Значение, разбивающее выборку на обучающую и тестовую

    # Вывод графика обучения
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Точность работы модели')
    plt.ylabel('Точность')
    plt.xlabel('Эпоха')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Сохранение модели
    model.save('cifar10_cnn.h5')
