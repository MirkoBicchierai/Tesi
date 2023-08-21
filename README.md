# Generazione di espressioni facciali 3D con una rete LSTM
Nel file requirements.txt si trovano tutte le libreire con le relative versioni e dipendenze necessarie per avvia il training della rete.

E' possibile scaricare il dataset CoMA da qui: https://coma.is.tue.mpg.de/ mentre per il dataset CoMA_Florence e disponibile qui: https://drive.google.com/drive/folders/14TLFQkWXPwujeApwpjbS15ZYA7_zirwl .

Per il dataset Coma sono state usate tutte le 12 label per ogni volto mentre per il dataset CoMA_Florence sono state usate solo 10 label su 70 (Cheeky, Confused, Cool, Displeased, Happy, Kissy, Moody, Rage, Sad2, Scream).

# Risultati

# Come usare la rete

Il dataset CoMA_Florence presenta dei file .obj per ogni frame che devono essere prima trasformati in .ply per il corretto funzionamento.

Per estrarre i landmark dal dataset CoMA_Florence usare il file get_animation_landmark_ply.py

Sulle sequenzze del dataset CoMA è stata effettuata un interpolazione e poi un campionato a 40 frame per ogni sequenza, il codice per farlo si trova nel file sampling_COMA.py. una volta eseguita tale operazione è possibile estrarre i landmark per ogni frame con il codice nel file get_landmark_COMA.py.

Il file main.py contiene il train loop della rete LSTMCell per la generazione dell'espressione 3D mentre nel file test.py la generazione vera e proprio della sequenza.

Nella cartella "Classification" si trova la rete LSTM per la classificazione delle sequenze generate.

