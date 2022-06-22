Project: Ergasia3__2020-2021

Eleftheria Ellina
A.M. : 1115201800228
Stylianos Psara
A.M. : 1115201800226

1. Ektelesi programmatos

EROTIMA A:
one by one time series:
python forecast.py -d nasdaq2007_17.csv -n 1
n series:
python forecast.py -d nasdaq2007_17.csv -n 4

EROTIMA B:
python predict.py -d nasdaq2007_17.csv -n 4 -mae 0.65

EROTIMA G:
python reduce.py -d nasdaq2007_17.csv -od input_D_encoded.csv -oq query_D_encoded.csv

2. Sxediastikes epiloges

Kata kirio logo exei xrisimopoiithei o kodikas apo ta tutorial gia ola ta erotimata kai exoun ginei allages gia tin sosti anagnosi ton arxeion mas.
Episis o diaxorismos tou train set kai test set exei alaxthei oste na litourgei sosta to training.
To fit ginete mia fora gia to sinolo kai san orisma pernei to training set sto opio exoun gini append ola ta komatia tou x_train mesa sto loop.
Kai ta tria montela exoun ekpedefti at home, to A kai B erotima me 20 xronoseires kai to G me 360. Exoume ston kodika sxoliasmena ta fit kai save model gia sinolo
kai kanoume apefthias load to idi trained montelo. Oi fakeloi me ta trained montela simperilamvanonte sta arxeia pou sas steilame: (model_A_at_home, model_B_at_home, model_G_at_home,
model_G_at_home_encoder -gia to predict ton xronoseiron gia to neo csv-).
Meso ton piramaton katalavame oti xreiastike na prosthesoume kapies entoles se kathe erotima gia tin veltistopiisi ton apotelesmaton, kapoies apo aftes itan kai to inverse_transform opou
xreiazotan. 
**Gia to erotima G pairnoume olokliro to arxeio nasdaq2007_17.csv gia input kai query kai o diaxorismos ginete entos programmatos.

Paradidonte etoima ta arxeia gia to D:
input_D_initial.csv query_D_initial.csv, me tis arxikes xronoseires
input_D_encoded.csv query_D_encoded.csv, me tis xronosires miomenis poliplokotitas





