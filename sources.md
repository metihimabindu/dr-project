# Data sources

## Cost (AWS Pricing Calculator, calculator.aws, read on 05-11-2026)
Instance m5.large, Linux, On-Demand, USD per hour, 1 instance, constant usage
- Mumbai (ap-south-1): 0.101
- Singapore (ap-southeast-1): 0.12
- Frankfurt (eu-central-1): 0.115
- Oregon (us-west-2): 0.096

## Carbon (gCO2e/kWh)
Ember Electricity Data Explorer, ember-energy.org, year 2025, read on 05-11-2026
- India: 671
- Singapore: 497
- Germany: 334
Link pattern: ember-energy.org/data/electricity-data-explorer/?entity=India&data=co2_intensity&fuel=total&tab=main&chart=single_year
(same link with entity=Singapore and entity=Germany)

- Oregon: 288.2
  Source: EPA eGRID2023 Summary Tables (summary_tables_rev2.xlsx), Table 1,
  subregion NWPP (WECC Northwest), total output emission rate CO2e = 635.267 lb/MWh,
  converted with 1 lb/MWh = 0.4536 g/kWh.
  Page link: https://www.epa.gov/egrid/summary-data (accessed 05-11-2026)
  Note: this is 2023 data, while the other three values are 2025 data.