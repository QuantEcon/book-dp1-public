# Request the api key from https://fred.stlouisfed.org/docs/api/api_key.html
#
# Nominal interest rate from https://fred.stlouisfed.org/series/GS1
# Real interest rate from https://fred.stlouisfed.org/series/WFII10
#

using FredData, DataFrames, CSV

api_key = "c410ad8953615114ae1fbb353f238033"

nominal_interest_id, real_interest_id = ["GS1", "WFII10"]

function download_data(id; key=api_key, date_start="1953-01-01", date_end="2023-08-01")
    f = Fred(key)
    pull_data = get_data(f, id; observation_start=date_start, observation_end=date_end)
    export_data = select(pull_data.data, :date, :value)
    header = ["DATE", id]
    CSV.write("data/"*id*".csv", export_data, header=vec(header))
end

download_data(nominal_interest_id) 

download_data(real_interest_id; date_start="2012-04-13") 