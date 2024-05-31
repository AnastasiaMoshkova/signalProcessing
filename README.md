# signal processing example

1. train.py - пример обучения простой MLP сети с управлением конфигурациями через Hydra и логированием результатов с WB
2. train_optuna.py - - пример обучения простой MLP сети с управлением конфигурациями через Hydra, логированием результатов с WB, поиском лучщих параметров с Optuna


Задание на семинар:
1. Напишите новый класс с моделью нейронной сети (сверточная или реккурентная сеть), запустите эксперимет с новым классом модели
2. Напишите новый класс с лосс функцией (MSE), запустите эксперимет с новым классом
3. Добавьте новый параметр для оптимизации с Optuna (например, learning rate)
4. Выберете chechpoint из папки с экспериментом, напишите функцию для получения метрик на тесте на выбранном chechpoint
5. 