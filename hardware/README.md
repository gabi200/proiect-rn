# Documentație hardware
## Arhitectură fizică
Sistemul hardware este controlat de un Arduino Nano, conectat la doua servomotoare de 180 grade. Placa de dezvoltare Arduino comunica prin serial cu PC-ul care ruleaza sistemul de inteligenta artificiala (SIA).

## Firmware
Firmware-ul primește de la SIA comenzi prin serial de forma 
`speed;signaling_mode;steering_mode`
unde 
`speed` - viteza in km/h
`signaling_mode` - modul de semnalizare (0 - oprit, 1 - semnalizare stanga, 2 - semnalizare dreapta, 3 - avarie) **(NEUTILIZAT)**
`steering_mode` - modul de viraj (0 - inainte, 1 - stanga, 2 - dreapta)
