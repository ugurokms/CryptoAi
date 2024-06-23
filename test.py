import DataPreprocessor as dp

# Örnek kullanım:
interval = '1d'
start_date = "01 Jan, 2021"
end_date = "10 Jul, 2023"
x, y = dp.get_data(interval, start_date, end_date)
