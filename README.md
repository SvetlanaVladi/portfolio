ПРОСТОЙ КАЛЬКУЛЯТОР
Составьте блок-схему работы калькулятора, который складывает, вычитает, умножает и делит 2 целых числа.
Калькулятор должен принимать в качестве входных данных 2 числа и оператор (сложение, вычитание, умножение, деление), выполнять требуемую операцию и выдавать ответ.

ИТОГОВАЯ БАЗА ДАННЫХ SQL
Создайте резервную копию базы данных. Из PhpMyAdmin это можно сделать через меню Экспорт (поставить галочку Структура таблиц), из командной строки через mysqldump <имя_базы_данных>.
Скопируйте вывод команды CREATE TABLE для всех таблиц в базе данных.
Приложите описание структуры таблиц и характерные SELECT запросы на выборку данных.

СТАТИСТИЧЕСКИЙ АНАЛИЗ
Изучите данные по динамике изменения величины прожиточного минимума в городе Москве:
https://video.ittensive.com/math-stat/data-6048-2020-06-29.utf.csv
Проверьте, что значения AveragePerCapita не распределены нормально (через QQ-Plot или любым другим методом).
Проведите дисперсионный анализ для серий Seniors и Children и установите, с каким p-уровнем значимости средние этих серий различаются.
Постройте регрессионную модель AveragePerCapita от Quarter, WorkingPopulation, Seniors и Children. Выпишите коэффициенты линейной модели и сделайте предсказание на второй квартал 2020 года.

ПРОГНОЗ ПОГОДЫ
Напишите программу на языке Python, которая выводит прогноз температуры в вашем городе на сегодня в формате
T +/- d
Где T - средняя температура в этот день за последние 9 лет, d - среднеквадратичное отклонение (по ГОСТу) температуры за последние 9 лет.
Для работы программы вам потребуется собрать данные по температуре в вашем городе в текущем месяце за последние 9 лет и организовать их в CSV или TSV формате следующего вида:
1;T1_1;T1_2;...;T1_9
2;T2_1;T2_2;...;T2_9
..
31;T31_1;T31_2;...;T31_9
Где T1_1 - температура в первый день месяца 10 лет назад, T1_2 - температура в первый день месяца 9 лет назад и т.д. T31_9 - температура в 31 день месяца год назад. Если в текущем месяце меньше 31 дня, то в файле с данными будет меньше строк - по числу дней в текущем месяце.
В поле Отчет скопируйте ваше программу, в качестве приложения загрузите CSV/TSV файл с данными.
Решение: подключим необходимые библиотеки(time, numpy, pandas). Откроем файл на чтение с помощью пандаса, обозначим что нет заголовка, разделитель, и сразу укажем какую колонку использовать в качестве индекса, по умолчанию по ней и будем работать. С помощью time получаем текущий день, и выводим результат с помощью f-строки, в которой сразу посчитаем среднее(это  будет наша температура на текущий день) и среднеквадратичное отклонение
