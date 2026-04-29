# Metódy strojového učenia pre včasnú predikciu geomagnetických búrok

Program: Hospodárska informatika
Vypracovala: Bc. Terézia Drengubiaková
Diplomová práca: : Metódy strojového učenia pre včasnú predikciu geomagnetických búrok
Vedúci diplomovej práce: doc. Ing. Peter Butka, PhD.
Konzultanti: Ing. Viera Krešňáková, PhD., RNDr. Šimon Mackovjak, PhD.

V tomto repozitári nájdete datasety, modely a kódy, ktoré sme použili pri tvorení diplomovej práce. 

Pred prvým spustením programu je dôležité nainštalovať príslušné knižnice. Možno tak urobiť v prostredí, ktoré podporuje .ipynb súbory. Knižnice nainštalujeme nasledovným príkazom:
```bash
!pip install pyarrow
!pip install keras
!pip install --upgrade tensorflow
!pip install --upgrade tensorflow-gpu
```
Zoznam balíkov potrebný na spustenie kódu je definovaný v každom zdrojovom kóde. Pred samotným spustením kódu je nutné použiť príslušný dataset. Repozitár obsahuje kód:
- prípravy dát (1_priprava_a_rozdelenie_dat)
- rozdelenia dát (1_priprava_a_rozdelenie_dat)
- modelovanie - obsahuje všetky modely, ktorými sme sa v tejto práci zaoberali, taktiež ich vyhodnotenie pomocou metrík (2_modelovanie)
- vizualizácie - obsahuje kód ku spusteniu vizualizácií (3_vizualizacie)

Dostupné datasety sa nachádzajú v priečinku 0_datasety:
- events_omni.csv
- test_omni.csv
- train_omni.csv
- dataset so stiahnutými dátami z HAPI sa nachádza v priečinku 1_priprava_a_rozdelenie_dat - DP_omni.csv



