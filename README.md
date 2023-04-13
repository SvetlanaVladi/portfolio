# portfolio
Задача1:
Возьмите данные по вызовам пожарных служб в Москве за 2015-2019 годы: https://video.ittensive.com/python-advanced/data-5283-2019-10-04.utf.csv Получите из них фрейм данных (таблицу значений). По этому фрейму вычислите среднее значение вызовов пожарных машин в месяц в одном округе Москвы, округлив до целых Примечание: найдите среднее значение вызовов, без учета года

Решение:
Подключим библиотеку pandas.
Загрузим данные в дата фрейм при помощи read_csv. Выведем шапку дата фрейма, чтобы убедиться, что все корректно загрузилось.
Выведем среднее значение вызовов пожарных машин в месяц, в одном округе Москвы, округлив до целых.


Задача2:
Получите данные по безработице в Москве: https://video.ittensive.com/python-advanced/data-9753-2019-07-25.utf.csv Объедините эти данные индексами (Месяц/Год) с данными из предыдущего задания (вызовы пожарных) для Центральный административный округ: https://video.ittensive.com/python-advanced/data-5283-2019-10-04.utf.csv Найдите значение поля UnemployedMen в том месяце, когда было меньше всего вызовов в Центральном административном округе.

Решение:
1 Подключим библиотеку pandas.
2 Загрузим данные в дата фрейм при помощи read_csv. Выведем шапку дата фреймов, чтобы убедиться, что все корректно загрузилось.
3 Из первого и второго набора данных удалим не нужные колонки.
4 Переименуем колонки в наборах данных.
5 Создадим список номеров индекса центрального округа.
6 Создадим список значений округа по ранее созданному списку индексов центрального округа.
7 Добавим в наш дата фрейм еще одну серию и скажем, что это индекс наших зон.
8 После этого назначим мульти индекс.
9 Сделаем срез данных по значению мульти индекса "Центральный".
10 Объединим данные по индексам.
11 Найдем минимальное количество вызовов и выведем их.

Задача3
Получите данные по безработице в Москве: https://video.ittensive.com/python-advanced/data-9753-2019-07-25.utf.csv Найдите, с какого года процент людей с ограниченными возможностями (UnemployedDisabled) среди всех безработных (UnemployedTotal) стал меньше 2%.

Решение:
Подключим библиотеку pandas. Чтобы выполнить фильтрацию заведем дополнительный столбец data[sum] в DateFrame в котором посчитаем отношение в процентах между столбцами "люди с ограниченными возможностями" (UnemployedDisabled) и "всего безработных" (UnemployedTotal). Установим индекс по колонке "Год". Отфильтруем все значения в этом столбце, которые меньше 2% и отсортируем их. Выведем самый первый индекс(со значением 0).  


Задача4:
Возьмите данные по безработице в городе Москва: video.ittensive.com/python-advanced/data-9753-2019-07-25.utf.csv Сгруппируйте данные по годам, и, если в году меньше 6 значений, отбросьте эти годы. Постройте модель линейной регрессии по годам среднего значения отношения UnemployedDisabled к UnemployedTotal (процента людей с ограниченными возможностями) за месяц и ответьте, какое ожидается значение процента безработных инвалидов в 2020 году при сохранении текущей политики города Москвы? Ответ округлите до сотых.

Решение:
Подключим библиотеку pandas, numpy и LinearRegression из библиотеки sklearn.linear_model. Загрузим данные в дата фрейм при помощи read_csv. Выведем шапку дата фрейма, чтобы убедиться, что все корректно загрузилось. Через лямбда функцию сформируем серию данных процент людей с ограниченными возможностями по отношению к общему числу безработных. Отфильтруем данные, отбросив все значения не удовлетворяющие условию "если в году меньше 6 значений". Получим новый фрейм данных data_avg сгруппировав данные во фрейме data по году, со средними значениями по проценту людей с ограниченными возможностями по отношению к общему числу безработных. Взьмем индекс данных (год) из data_avg преобразуем его к таблице по средствам reshape. Аналогично поступим с серией данных значений по проценту людей с ограниченными возможностями по отношению к общему числу безработных в data_avg. Вызовем модель LinearRegression() и вгрузим в нее данные по средством fit(x, y). Выведем предсказания (model.predict(np.array(2020).reshape(1, 1)).

Задача5:
Изучите API Геокодера Яндекса tech.yandex.ru/maps/geocoder/doc/desc/concepts/input_params-docpage/ и получите ключ API для него в кабинете разработчика. Выполните запрос к API и узнайте долготу точки на карте (Point) для города Самара. Внимание: активация ключа Геокодера Яндекса может занимать несколько часов (до суток)

Решение:
Подключаем необходимые библиотеки: json, pandas, requests. Делаем запрос, с параметрами согласно API Геокодера Яндекса, формируя поисковую строку. Через json.loads загрузим ответ, выведем нужные данные из загруженного json.

Задача6:
Получите данные по котировкам акций со страницы: mfd.ru/marketdata/?id=5&group=16&mode=3&sortHeader=name&sortOrder=1&selectedDate=01.11.2019 и найдите, по какому тикеру был максимальный рост числа сделок (в процентах) за 1 ноября 2019 года.

Решение:
Подключим библиотеки requests, bs4, pandas.
Сделаем get запрос к странице с котировками. Ответ загрузим в BeautifulSoup для дальнейшего поиска нужных данных. Найдем таблицу с тегом mfd-table и назначим ее как переменную table. Переберем все значения tr найденные в table. По средствам get_text(strip=True) в каждом найденном td в tr уберем лишние пробелы. Проверяем tr на условие, что он не пустой (len(tr)>0). Добавляем не пустые в tr. Переименуем колонки для удобства. Отфильтруем наши данные, оставим только те, по которым были сделки.
Чтобы мы могли работать с числовыми данными, удалим '%' и заменим тире на знак минуса, а также укажим тип данных как float в колонке "Сдел.%". Отсортируем по данным из колонки "Сдел.%" в порядке убывания (ascending=False). Выведем только первое значение в полученных данных, это и будет ответ на поставленную задачу.

Задача7:
Используя парсинг данных с маркетплейса beru.ru, найдите, на сколько литров отличается общий объем холодильников Саратов 263 и Саратов 452? Для парсинга можно использовать зеркало страницы beru.ru с результатами для холодильников Саратов по адресу: video.ittensive.com/data/018-python-advanced/beru.ru/

Решение:
Подключим библиотеку requests, bs4.
Сделаем get запрос к странице холодильниками и загрузим в BeautifulSoup.
Получим ссылки со страницы, поиском всех тегов начинающийся на "a".
Выведем все содержимое BeautifulSoup и ищем сарартов 263, находи тег с классом который будет указывать на название холодильника и ссылку на его страницу.
Собираем все ссылки с этим классом.
Теперь с помощью перебора ссылок найдем ту которая указывает на Саратов 263 и Саратов 452, обозначим их как link_263 и link_452.
По отобранным ссылкам через BeautifulSoup найдем класс который указывает на общий объем холодильника.
Берем нужный нам по счету и через генератор из текста возвращаем только цифры, это и будут наши объемы.
Вычитаем объем 263 из 452 и получаем искомые данные.

Задача8:
Соберите данные о моделях холодильников Саратов с маркетплейса beru.ru: URL, название, цена, размеры, общий объем, объем холодильной камеры. Создайте соответствующие таблицы в SQLite базе данных и загрузите полученные данные в таблицу beru_goods. Для парсинга можно использовать зеркало страницы beru.ru с результатами для холодильников Саратов по адресу: video.ittensive.com/data/018-python-advanced/beru.ru/

Решение:
Подключим библиотеку requests, bs4, sqlite3, pandas.
Сделаем get запрос к странице холодильниками и загрузим в BeautifulSoup.
Найдем класс тега который указывает на ссылки холодильников и соберем по нему все ссылки на холодильники.
Отберем только ссылки "Саратов".
Подключаемся в БД и создаем таблицу beru_goods в базе при помощи db.execute().
Соберем информацию для загрузки в БД, загружая по очереди данные в BeautifulSoup из ранее полученных ссылок, собирая оттуда по заранее найденным классам название, цену, объем и размер.
Для нахождения данных по общему объему и объему морозильный камеры, дополнительно переберем данные по объёму и отберем те которые подходят по условиям, а также получим только числовое значение объема.
Полученные данные загрузим в БД.
Загрузим из БД через pd.read_sql_query все данные, выведем их, убедимся, что все прошло корректно.
Закроем соединение с БД.

Задача9:
Загрузите данные по ЕГЭ за последние годы https://video.ittensive.com/python-advanced/data-9722-2019-10-14.utf.csv выберите данные за 2018-2019 учебный год. Выберите тип диаграммы для отображения результатов по административному округу Москвы, постройте выбранную диаграмму для количества школьников, написавших ЕГЭ на 220 баллов и выше. Выберите тип диаграммы и постройте ее для районов Северо-Западного административного округа Москвы для количества школьников, написавших ЕГЭ на 220 баллов и выше.

Решение:
Подключаем библиотеки pandas, matplotlib Загружаем данные. преобразуем данные, чтобы избежать дубликатов и длинных названий. Устанавливаем район как категория для ускорения сортировки. Выставляем индекс по году и получаем с помощью фильтрации данные за промежуток 2018-2019 года. После этого сбрасываем индексы. Строим две круговые диаграммы в одну строку.

Задание10:
Загрузите данные по итогам марафона https://video.ittensive.com/python-advanced/marathon-data.csv Приведите время половины и полной дистанции к секундам. Найдите, данные каких серии данных коррелируют (используя диаграмму pairplot в Seaborn). Найдите коэффициент корреляции этих серий данных, используя scipy.stats.pearsonr. Постройте график jointplot для коррелирующих данных.

Решение:
Подключаем библиотеки pandas, matplotlib, seaborn, scipy.stats Загружаем данные. Переводим время в секунды, предварительно конвертировав его в числовой вид. Строим парный график по всем парным числовым значениям, чтобы посмотреть какие данные коррелируют друг с другом. Рассчитываем коэффициент Пирсона и строим joint-график.

Задание11:
Используя данные индекса РТС за последние годы https://video.ittensive.com/python-advanced/rts-index.csv постройте отдельные графики закрытия (Close) индекса по дням за 2017, 2018, 2019 годы в единой оси X. Добавьте на график экспоненциальное среднее за 20 дней для значения Max за 2017 год. Найдите последнюю дату, когда экспоненциальное среднее максимального дневного значения (Max) в 2017 году было больше, чем соответствующее значение Close в 2019 году (это последнее пересечение графика за 2019 год и графика для среднего за 2017 год).

Решение:
Подключим библиотеку pandas, matplotlib.pyplot, plotly.graph_objects. Загрузим данные. Переведем формат даты из отечественного в западный при помощи pd.to_datetime с параметром dayfirst=True. Для того чтобы совместить все 4 графика на одной координатной плоскости Х, добавим новую колонку "Day" для номера дня в году. Сформируем список из дат, начальной будет являться min(data["Date"]), а конечной max(data["Date"]). Назначим индекс по Date и переиндексируем по нашему списку dates, при этом отсутствующие значения заполним предыдущими значениями. Добавим еще одну серию данных, день года, которая понадобится нам для совмещения данных на оси Х. Назначим имя индекса которое потерялось при переиндексации по списку дат и отсортируем по датам, развернув тем самым данные в нужном порядке. Подготовим два набора данных, по которым в последующем будем искать пересечение графиков. Построим отдельные графики закрытия (Close) индекса по дням за 2017, 2018, 2019 годы в единой оси X, а также график экспоненциальное среднее за 20 дней для значения Max за 2017 год. Добавим легенду к графикам и выведем, визуально наблюдаем пересечение. Найдем точную дату пересечения, отфильтровав данные до момента когда данные закрытия 2019 станут больше данных экспоненциального среднего значения Max за 2017 год. Зададим индекс по "Date", отсортируем по убыванию и выведем 1й индекс, это и будет искомое значение. 

Задача12:
Изучите набор данных по объектам культурного наследия России (в виде gz-архива): https://video.ittensive.com/python-advanced/data-44-structure-4.csv.gz и постройте фоновую картограмму по количеству объектов в каждом регионе России, используя гео-данные https://video.ittensive.com/python-advanced/russia.json Выведите для каждого региона количество объектов в нем. Посчитайте число объектов культурного наследия в Татарстане.

Решение:
 Подключаем библиотеки matplotlib, geopandas, pandas, descartes. Так как файлы с данными имеют большой размер, то загрузим все данные локально и будем их использовать как находящиееся на нашем носителе информации. Для анализа данных будем использовать и загружать в DateFrame только те данные, которые будут нужны для построения карты и подсчета числа объектов. Подсчитаем количество объектов культурного наследия в каждом регионе. Загрузим геоданные и приведем их сразу к меркатору. Унифицируем названия одного и того же региона указанного в разных DateFrame. Объеденим полученные и исправленные наборы данных. объединение проводим по колонке "Регион". Создаем холст и область отрисовки. На каждый отрисованный регион наносим в центре количество объектов культурного наследия. Так как карта сильно растягивается обрезаем на ней два региона - Калининградскую область и Чукотку. Итоговой операцией просто выводим для пользователя количество объектов культурного наследия в Татарстане.
 
 Задание13:
 Используя данные по посещаемости библиотек в районах Москвы https://video.ittensive.com/python-advanced/data-7361-2019-11-28.utf.json постройте круговую диаграмму суммарной посещаемости (NumOfVisitors) 20 наиболее популярных районов Москвы. Создайте PDF отчет, используя файл https://video.ittensive.com/python-advanced/title.pdf как первую страницу. На второй странице выведите итоговую диаграмму, самый популярный район Москвы и число посетителей библиотек в нем.
 
 Решение:
 Подключаем библиотеки: requests, json, pandas, matplotlib, seaborn, а также отдельные библиотеки для работы с PDF-файлами: from reportlab.pdfgen import canvas, from reportlab.lib import pagesizes, from reportlab.pdfbase import pdfmetrics, from reportlab.pdfbase.ttfonts import TTFont, from reportlab.lib.utils import ImageReader, from PyPDF2 import PdfFileMerger, PdfFileReader. Загружаем данные в формате json, затем передадим их в DateFrame и заполняем отсутствующие данные нулями.. Определяем, что каждый набор данных это вложенный словарь. Для группировки по району нужно извлечь название района из этой структуры. Используя функцию extract_district извлекаем эти данные. Ищем значение района, находим в нем поле "District", приводим все найденные значения в список и возвращаем первое найденное значение. По факту будет вытащено первое значение "District" из словаря "ObjectAddress". После получения района группируем данные и сортируем их по числу посетителей в порядке убывания. Строим итоговую круговую диаграмму из 20 самых популярных районов. На самом графике делать надписи не будем - перенесем их в легенду и выведем ее справа от диаграммы. Сохраняем график в файл для дальнейшей вставки в отчет. Формируем отчет: задаем шрифт, холст с размерами. Нанесем данные по читателям библиотек и укажем, что номер этой страницы 2. Вставляем нашу сохраненную диаграмму из файла. Дополнительно выводим самый популярный район (это первая строка в нашем отсортированном списке) и количество читателей в нем. Объединяем титульную страницу и созданный отчет через PdfFileManager.
 
 Задача14:
 Сгенерируйте PDF документ из списка флагов и гербов районов Москвы: https://video.ittensive.com/python-advanced/data-102743-2019-11-13.utf.csv На каждой странице документа выведите название геральдического символа (Name), его описание (Description) и его изображение (Picture). Для показа изображений используйте адрес https://op.mos.ru/MEDIA/showFile?id=XXX где XXX - это значение поля Picture в наборе данных. В случае возникновения проблем с загрузкой изображений с op.mos.ru можно добавить в код настройку для форсирования использования дополнительных видов шифрования в протоколе SSL/TLS. requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL:@SECLEVEL=1
 
 Решение:
 Подключим библиотеки pandas, requests, pdfkit. Сконфигурируем pdfkit и загрузим данные в дата фрейм. Сформируем HTML код (станицу) для создания отчета, в котором через for переберем все строки из фрейма и получим все необходимые страницы отчета, так как туда будут подставляться изображение символа, его название и описание.
Введем условие, что если это первая станица, то выводим обычный стиль заголовка, если нет то выводим style="page-break-before:always", что делает разрыв раздела с каждой новой страницы, чтобы каждый знак был на отдельной странице. Прописываем опции для pdfkit, размер страницы и нумерацию. Из HTML кода при помощи pdfkit собираем PDF документ отчета.

Задача15:
Используя данные по активностям в парках Москвы https://video.ittensive.com/python-advanced/data-107235-2019-12-02.utf.json Создайте PDF отчет, в котором выведите:
Диаграмму распределения числа активностей по паркам, топ10 самых активных
Таблицу активностей по всем паркам в виде Активность-Расписание-Парк

Решение: 
Подключаем библиотеки: requests, json, pandas, matplotlib, из io import BytesIO, binascii, pdfkit. Загрузим и сформируем DateFrame только из нужных нам колонок. Получим значение поля value из словаря NameOfPark и занесем это значение в колонку NameOfPark. Переименуем колонки для лучшего восприятия данных в отчете. Формируем диаграммуактивности по паркам. Создаем холст. Группируем данные по паркам, отсортируем их по убыванию по колонке "Активность". Чтобы не сохранять локально файлы с изображениями будем использовать библиотеку binascii для хранения изображения в памяти в формате UTF-8. Чтобы pandas не ограничивал длину данных зададим настройки через set_options библиотеки pandas. Через html строку создаем параметры вывода, затем генерируем pdf-отчет, используя данные из html-строки.

Задача16:
Соберите отчет по результатам ЕГЭ в 2018-2019 году, используя данные https://video.ittensive.com/python-advanced/data-9722-2019-10-14.utf.csv и отправьте его в HTML формате по адресу support@ittensive.com, используя только Python. В отчете должно быть:
1 общее число отличников (учеников, получивших более 220 баллов по ЕГЭ в Москве),
2 распределение отличников по округам Москвы,
3 название школы с лучшими результатами по ЕГЭ в Москве. Диаграмма распределения должна быть вставлена в HTML через data:URI формат (в base64-кодировке). Дополнительно: приложите к отчету PDF документ того же содержания (дублирующий письмо).

Решение:
Подключим pandas, matplotlib, pdfkit, из io import BytesIO, binascii, smtplib, из email import encoders, из email.mime.text import MIMEText, из email.mime.base import MIMEBase, из email.mime.multipart import MIMEMultipart Загружаем данные и выделяем результат за 2018-2019 года. Отсортируем данные, чтобы найти лучшие школы по результатам ЕГЭ. Сгруппируем данные по административным округа, при этом убирая в названии округа все слова кроме первого, чтобы подписи на графиках были более читабельны. Подсчитаем общее количество отличников - оно будет нужно для дальнейших расчетов. Создаем холст на котором создадим список секторов. Создаем круговую диаграмму по округам с параметрами вывода и легендой. Два округа с самыми маленькими значениямисделаем выносными для лучшей визуализации. Чтобы сохранить изображение в памяти будем использовать библиотеку binascii для хранения изображения в памяти в формате UTF-8.Параллельно сохраним изображение локально. Чтобы pandas не ограничивал длину данных зададим настройки через set_options библиотеки pandas. Сформируем html - отчет со всеми необходимыми данными. Сконфигурируем pdf- отчет с указанными настройками и сохраним в pdf-файле. Для отправки письма по электронной почтесоздаем объект letter = MIMEMultipart(). Задаем поля у этого объекта. В тело письма прикрепим html-документ и после этого вложим в письмо наш отчет. Подключаем smpt-server, указав имя пользователя (отправителя) и его пароль. После этого отправляем писсьмо и закрываем smpt-server.
