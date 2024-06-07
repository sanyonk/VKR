# Сервис по суммаризации и извлечению метрик из новостных статей

## О сервисе
Сервис по суммаризации и извлечению метрик из новостных статей разработан для аналитиков безопасности, чтобы они могли за короткий промежуток времени извлекать метрики из текста, при этом сам текст новости читать не требуется. Сервис разработан для новостных статей на английском языке. 

## Как запускать сервис
Для того, чтобы запустить сервис, нужно перейти в директорию, где лежит файл ```run news_reader_app.py```, а далее запустить команду, которая представлена ниже:
```sh
streamlit run news_reader_app.py
```
После запуска команды должно появиться окно для ссылки, чтобы вставить одну ссылку на новость. Далее нажать кнопку "Прочитать новость" и метрики начнут выделяться из текста.

## Ссылки на новости, чтобы попробовать сервис
Ниже представлены ссылки на новости о кибератаках на английском языке, которые можно вставить в сервис и получить метрики из текста:
```sh
https://thehackernews.com/2024/02/us-doj-dismantles-warzone-rat.html
https://thehackernews.com/2024/03/exit-scam-blackcat-ransomware-group.html
https://thehackernews.com/2024/02/authorities-claim-lockbit-admin.html
```