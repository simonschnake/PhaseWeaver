# CRISP Reconstruction: Algorithmus und Parameter

Diese Notiz dokumentiert den Rekonstruktionsablauf aus dem echten CRISP-Code aus
`/Users/schnakes/Downloads/crisp-reconstruct-main.zip`.

Ausgewertet wurden vor allem:

- `src/EqFctCrispReconstruction.cc`
- `src/EqFctCrispReconstruction.h`
- `src/functions.cc`
- `src/functions.h`
- `tests/test_EqFctCrispReconstruction.cc`
- `tests/test_functions.cc`

Der Server rekonstruiert aus gemessenem CRISP-Formfaktorbetrag ein longitudinales
Stromprofil. Eingang ist der quadrierte Betrag des Formfaktors, also `|F|^2`, zusammen
mit Frequenzen, Standardabweichungen, Detektionsgrenzen und Bunch-Ladung.

## Wichtige Default-Werte

| Parameter | Wert im Code | Bedeutung |
| --- | ---: | --- |
| `input_form_factor_size` | `240` | nominelle Groesse der ZMQ-Eingangsarrays |
| minimale akzeptierte Eingangslaenge | `100` | kuerzere Frequenzarrays werden verworfen |
| `num_output_points_` | `1024` | Default-Laenge des finalen Stromprofils |
| erlaubte Output-Laenge | `128 ... 1_048_576` | Eingaben werden auf die naechste Zweierpotenz gerundet |
| `max_frequency_` | `2048 THz` | Default-Maximalfrequenz der extrapolierten Skala |
| erlaubte Maximalfrequenz | `60 ... 100000 THz` | Werte darunter/darueber werden geklemmt |
| Frequenzschritt | `dnu = 2 * max_frequency / num_output_points` | bei Defaults: `4 THz` |
| Zeitschritt | `dt = 0.5 / max_frequency` | bei Defaults: ca. `0.244 fs` |
| Zeitfenster | `num_output_points * dt` | bei Defaults: ca. `250 fs` |
| Zeitstart | `-0.5 * num_output_points * dt` | bei Defaults: ca. `-125 fs` |
| `detection_limit_fudge_` | `1.0` | Faktor auf die Detektionsgrenze |
| erlaubter Detection-Fudge | `0.0 ... 10.0` | andere Werte werfen Fehler |
| Minimum-Ladung | `10 pC` | darunter: `No beam` |
| Iterationsmaximum | `20` | harte Grenze fuer die iterative Rekonstruktion |
| Formfaktor-Fehlerband fuer Iteration | `20 %` von `|F|` | aus interpoliertem Betrag abgeleitet |
| Mindestfehler fuer Eingang `|F|^2` | `20 %` von `|F|^2` | Standardabweichung wird mindestens darauf gesetzt |
| Savitzky-Golay-Filter | Ordnung `3`, links `4`, rechts `4` | 9-Punkt-Kernel |
| Low-Frequency-Fit | `sigma` in `[1e-3, 1000] THz`, rel. Praezision `1e-4`, Startwert `4.0` | Gaussian-Fit auf `|F|^2` |
| High-Frequency-Fit | letzter gueltiger Messpunkt | analytische Gaussian-Breite |

## Eingangsdaten

Die DOOCS/ZMQ-Properties sind:

- `INPUT_FREQUENCIES`: Frequenzen in `THz`
- `INPUT_FFSQ`: gemessener quadrierter Formfaktorbetrag `|F|^2`
- `INPUT_FFSQ_STD`: Standardabweichung von `|F|^2`
- `INPUT_FFSQ_DETECTION_LIMIT`: Detektionsgrenze fuer `|F|^2`
- `CHARGE`: Ladung in `nC`
- `FAKE_CHARGE`: optionaler Ersatzwert in `nC`; wird verwendet, wenn `> 0`

Beim Trigger `interrupt_usr1(1)` werden Timestamp und Event-ID aus dem Bunch-Pattern
uebernommen. Danach laeuft `perform_reconstruction()`.

## Ablauf der Rekonstruktion

### 1. Ladung pruefen

Die verwendete Ladung ist:

```text
charge = FAKE_CHARGE, falls FAKE_CHARGE > 0
charge = CHARGE sonst
```

Wenn `charge < 10 pC`, bricht die Rekonstruktion mit `No beam` ab.

### 2. Eingangsfrequenzen sortieren

Die Frequenzskala wird neu importiert, wenn sich die Array-Laenge geaendert hat.

Gueltigkeitsregeln:

- Laenge muss mindestens `100` sein.
- Laenge darf nicht groesser sein als `uint16_t::max`.
- Die erste Frequenz muss `> 0 THz` sein.
- Die Frequenzen werden aufsteigend sortiert.
- Die gleiche Sortierung wird auf `INPUT_FFSQ`, `INPUT_FFSQ_STD` und
  `INPUT_FFSQ_DETECTION_LIMIT` angewendet.

### 3. Ungueltige Punkte entfernen

Zunaechst wird eine Maske mit allen Punkten auf `true` angelegt. Dann:

- Die ersten zwei Punkte werden immer verworfen.
- Die letzten zwei Punkte werden immer verworfen.
- Fuer jeden inneren Punkt `i` gilt:
  - Wenn `ffsq[i]` nicht endlich ist, werden `i - 1`, `i`, `i + 1` verworfen.
  - Wenn `ffsq[i] < detection_limit[i] * detection_limit_fudge`, werden ebenfalls
    `i - 1`, `i`, `i + 1` verworfen.
- Sobald `16` aufeinanderfolgende schlechte Maskenwerte gefunden werden, wird ab dort
  der komplette Rest verworfen.

Danach werden Frequenz, `|F|^2`, Standardabweichung und Detektionsgrenze mit derselben
Maske reduziert. `MAX_INPUT_FREQUENCY` ist anschliessend die letzte verbleibende
Frequenz.

### 4. Eingangswerte bereinigen

Nach der Maskierung:

- `|F|^2` wird auf `[0, 1]` geklemmt.
- `INPUT_FFSQ_STD[i]` wird mindestens auf `0.2 * ffsq[i]` gesetzt.

Der weitere Algorithmus nutzt diese Standardabweichung im aktuellen Code nicht direkt
fuer die iterative Modulus-Korrektur. Dort wird spaeter ein neues Fehlerband aus dem
interpolierten `|F|` erzeugt.

### 5. Gaussian-Extrapolation nach niedrigen und hohen Frequenzen

Der Code baut eine Zwischenfrequenzskala `INTERMEDIATE_FREQUENCIES` und ein
Zwischen-`|F|^2` `INTERMEDIATE_FFSQ`.

#### 5.1 Low-Frequency-Sigma

Fuer niedrige Frequenzen wird ein Gaussian-Modell

```text
ffsq(nu) = exp(-0.5 * (nu / sigma_low)^2)
```

an Eingangsdaten angepasst.

Punktwahl:

- Bei weniger als `5` gueltigen Punkten: Fehler.
- Bei `5 ... 19` Punkten: alle Punkte.
- Ab `20` Punkten: Punkte mit Index `5 ... 19`, also 15 Punkte.

Optimierung:

- `sigma` in `[1e-3, 1000] THz`
- relative Praezision `1e-4`
- Startwert `4.0`
- Zielfunktion: Summe der quadratischen Abweichungen zwischen gemessenem `ffsq` und
  Gaussian.

#### 5.2 High-Frequency-Sigma

Fuer hohe Frequenzen wird nur der letzte gueltige Messpunkt verwendet:

```text
sigma_high = sqrt(-nu_last^2 / log(ffsq_last))
```

Falls `ffsq_last <= 0`, wird fuer den Fit `ffsq_last = 1e-30` verwendet.

Das spaetere High-Frequency-Modell ist:

```text
ffsq(nu) = exp(-(nu / sigma_high)^2)
```

Auffaellig: Low-Frequency nutzt den Faktor `-0.5`, High-Frequency nutzt `-1.0`.

#### 5.3 Punkte vor der ersten Messfrequenz

Der Ziel-Frequenzschritt ist:

```text
dnu = 2 * max_frequency / num_output_points
```

in `THz`. Fuer den Front-Bereich wird `dnu_front` initial auf `dnu` gesetzt und so lange
halbiert, bis vor der ersten gemessenen Frequenz mindestens 4 Punkte liegen:

```text
while first_frequency / dnu_front < 4:
    dnu_front *= 0.5
```

Dann werden Frequenzen `nu = i * dnu_front` von `0` bis kleiner als die erste gemessene
Frequenz eingefuegt. Der Wert ist:

```text
ffsq = exp(-0.5 * nu^2 / sigma_low^2)
```

### 5.4 Gemessene Punkte und High-Frequency-Tail

Nach den Low-Frequency-Punkten werden alle gefilterten Messpunkte unveraendert
angehaengt. Der Index direkt danach wird als `idx_extrap_start` gespeichert.

Dann wird hinten erweitert:

1. Zuerst werden genau `10` Punkte mit dem Abstand der letzten beiden gemessenen
   Frequenzen angehaengt.
2. Danach wird mit dem regulaeren `dnu` weiter bis kleiner als `max_frequency`
   aufgefuellt.
3. Alle diese Punkte nutzen:

```text
ffsq = exp(-nu^2 / sigma_high^2)
```

### 6. Savitzky-Golay-Glaettung

Auf `INTERMEDIATE_FFSQ` wird ein Savitzky-Golay-Filter angewendet:

- Ordnung `3`
- `4` Punkte links
- `4` Punkte rechts
- Faltung mit Armadillo `conv(..., "same")`
- Die ersten `4` und letzten `4` Samples bleiben ungefiltert.
- Danach wird wieder auf `[0, 1]` geklemmt.

Wichtig: Die Funktionssignatur nimmt `idx_extrap_start` entgegen und der Header-Kommentar
beschreibt, dass die High-Frequency-Extrapolation ausgenommen werden soll. Die aktuelle
Implementation ignoriert diesen Parameter aber komplett. Faktisch wird alles ausser den
ersten und letzten vier Samples geglaettet.

### 7. Interpolation auf aequidistante positive Frequenzen

Aus `INTERMEDIATE_FREQUENCIES` und `INTERMEDIATE_FFSQ` wird ein 1D-Interpolator gebaut.
Interpoliert wird nur die positive Haelfte:

```text
half_num_output_points = num_output_points / 2
nu[i] = i * dnu, i = 0 ... half_num_output_points - 1
ffabs[i] = sqrt(interpolated_ffsq(nu[i]))
ffabs_error[i] = 0.2 * ffabs[i]
```

Bei Default-Werten:

- `num_output_points = 1024`
- `half_num_output_points = 512`
- `max_frequency = 2048 THz`
- `dnu = 4 THz`
- positive Frequenzen: `0, 4, 8, ..., 2044 THz`

Der Index `idx_start_upper_extrapolation` wird danach als erster interpolierter
Frequenzpunkt groesser als `MAX_INPUT_FREQUENCY` bestimmt. Ab diesem Index gelten Punkte
als High-Frequency-Extrapolation und werden in der iterativen Modulus-Korrektur
geschuetzt.

### 8. Kramers-Kronig-Phase berechnen

Die Phase wird auf der interpolierten positiven Frequenzskala aus dem Betrag berechnet.
Zuerst wird die erste Stelle gesucht, an der `ffabs <= 0`. Nur der davorliegende
nichtverschwindende Teil geht in die Summation ein.

Wenn weniger als `8` nichtverschwindende Punkte vorhanden sind, wird abgebrochen.

Fuer jeden Punkt `i`:

```text
phase[i] = (2 / pi) * f_i * dnu_THz *
           sum_{k != i} log(ff_k / ff_i) / (f_i^2 - f_k^2)
```

Wenn `ff_k / ff_i` numerisch `0` wird, wird dieser Summand uebersprungen.

Alle Punkte ab dem ersten Null-/Nichtpositivpunkt bekommen die letzte berechnete Phase:

```text
phase[i >= num_nonzero_points] = phase[num_nonzero_points - 1]
```

### 9. Komplexen Formfaktor bauen

Aus `ffabs` und Kramers-Kronig-Phase wird ein komplexer positiver Halb-Formfaktor:

```text
F[i] = polar(ffabs[i], phase[i])
```

Dann wird der Bereich auf doppelte Laenge erweitert, indem konjugiert gespiegelte Werte
angehaengt werden:

```text
F_full = [F[0], F[1], ..., F[M-1], conj(F[M-1]), ..., conj(F[1]), conj(F[0])]
```

Damit hat `F_full` die Laenge `num_output_points`.

### 10. Kramers-Kronig-Startprofil

Der komplexe Formfaktor wird per iFFT in den Zeitbereich transformiert:

```text
profile = real(ifft(F_full))
```

Dann wird das Profil zyklisch so rotiert, dass das Maximum auf dem zentralen Index liegt.
Bei gerader Laenge ist das der linke der beiden mittleren Indizes:

```text
center_index = (N - 1) / 2
```

Dieses Profil wird als `KRAMERS_KRONIG_CURRENT_PROFILE` publiziert, nach Skalierung auf
Strom in Ampere.

### 11. Iterative Rekonstruktion

Die Iteration startet mit dem Kramers-Kronig-Profil.

Maximal werden `20` Iterationen ausgefuehrt. Pro Iteration:

1. Im Zeitbereich wird `isolate_positive_maximum()` angewendet.
   - Das groesste Element wird gesucht.
   - Falls kein positives Maximum existiert, wird das gesamte Profil auf `0` gesetzt.
   - Rechts vom Maximum wird ab dem ersten Wert `<= 0` alles auf `0` gesetzt.
   - Links vom Maximum wird bis zum ersten Wert `<= 0` alles ausserhalb auf `0` gesetzt.
   - Effekt: Nur der zusammenhaengende positive Peak um das Maximum bleibt erhalten.

2. Die ersten vier Iterationsprofile werden in `CURRENT_PROFILE_1` bis
   `CURRENT_PROFILE_4` publiziert.

3. Es wird eine FFT des Profils berechnet:

```text
ff = fft(profile)
```

4. Der Formfaktor wird auf DC normiert:

```text
if ff[0] != 0:
    ff[:] *= 1 / ff[0]
```

5. Fuer die positive Haelfte und separat fuer die negative Haelfte wird der Betrag
   kontrolliert. Fuer jeden Index `i < idx_start_upper_extrapolation`:

```text
if abs(abs(ff[i]) - ref_ffabs[i]) > ref_ffabs_error[i]:
    ff[i] = polar(ref_ffabs[i], arg(ff[i]))
    num_replaced += 1
```

Punkte ab `idx_start_upper_extrapolation` bleiben unangetastet. Das ist der extrapolierte
Hochfrequenzbereich jenseits der letzten gueltigen Messfrequenz.

6. Aus dem korrigierten komplexen Formfaktor wird wieder ein zentriertes Zeitprofil
   berechnet:

```text
profile = real(ifft(ff))
center_maximum(profile)
```

7. Danach wird `cnt` erhoeht.

8. Wenn in dieser Iteration kein einziger Modulus ersetzt wurde (`num_replaced == 0`),
   endet die Iteration vorzeitig.

Die gestoppte Iterationszahl wird als `NUM_ITERATIONS` gespeichert.

### 12. Finales Profil und Skalierung

Nach der Iteration wird nochmals `isolate_positive_maximum()` auf das Profil angewendet.
Dann werden `CURRENT_PROFILE` und `CURRENT_PROFILE.HOLD` aktualisiert.

Die Skalierung auf Ampere passiert so:

```text
current_A_sum = charge / dt
profile_scaled = profile * (current_A_sum / sum(profile))
```

Dabei ist `dt = 0.5 / max_frequency`. Die Summe der Samples entspricht also dem Strom,
der zur Ladung pro Zeitschritt passt.

### 13. Zeitachse und Skalarparameter

Die Zeitachse `OUTPUT_TIMES` wird in `fs` aufgebaut:

```text
dt_fs = dt / 1 fs
t_start = -0.5 * num_output_points * dt_fs
time[i] = t_start + i * dt_fs
```

Aus `CURRENT_PROFILE` werden berechnet:

- `PEAK_CURRENT`: Maximum des Stromprofils in `A`
- `FWHM`: volle Halbwertsbreite in `fs`
- `RMS_WIDTH`: RMS-Breite in `fs`
- `SKEWNESS`: biased skewness

Mean, RMS und Skewness werden nach Momenten des Stromprofils berechnet:

```text
mean = sum(t_i * I_i) / sum(I_i)
rms = sqrt(sum((t_i - mean)^2 * I_i) / sum(I_i))
skewness = sum((t_i - mean)^3 * I_i) / sum(I_i) / rms^3
```

## Rekonstruktion in Kurzform

```text
Trigger
  -> Ladung lesen, No-beam-Schwelle pruefen
  -> Frequenzen sortieren und Messarrays mit derselben Permutation sortieren
  -> ungueltige Punkte anhand Detektionslimit/NaN/Nachbarschaft entfernen
  -> |F|^2 auf [0, 1] klemmen, Fehler auf mindestens 20 % setzen
  -> Low-Frequency-Gaussian fitten
  -> High-Frequency-Gaussian aus letztem Messpunkt bestimmen
  -> nach 0 THz und bis max_frequency extrapolieren
  -> Savitzky-Golay glaetten
  -> auf aequidistante positive Frequenzskala interpolieren, |F| = sqrt(|F|^2)
  -> Kramers-Kronig-Phase berechnen
  -> komplexen symmetrischen Formfaktor bauen
  -> iFFT -> Kramers-Kronig-Startprofil
  -> bis zu 20 Iterationen:
       Zeitbereich: nur positiven Hauptpeak behalten
       FFT und DC-Normalisierung
       Frequenzbereich: gemessenen Betrag ausserhalb 20 %-Band erzwingen
       iFFT und Maximum zentrieren
       Stop, wenn keine Betragsersetzung noetig war
  -> final positiven Hauptpeak isolieren
  -> auf Ladung/Zeitschritt normieren
  -> Zeitachse und Profilkennzahlen berechnen
```

## Outputs

Wichtige rekonstruierte oder diagnostische Properties:

- `CURRENT_PROFILE`: finales Stromprofil in `A`
- `CURRENT_PROFILE.HOLD`: letztes gutes finales Stromprofil
- `CURRENT_PROFILE_1 ... CURRENT_PROFILE_4`: die ersten vier Iterationsprofile
- `KRAMERS_KRONIG_CURRENT_PROFILE`: Startprofil nach Kramers-Kronig-Phase
- `KRAMERS_KRONIG_PHASE`: rekonstruierte Phase
- `OUTPUT_TIMES`: Zeitachse in `fs`
- `INTERMEDIATE_FREQUENCIES`: Frequenzskala nach Extrapolation
- `INTERMEDIATE_FFSQ`: extrapolierter und geglaetteter quadrierter Formfaktor
- `INTERPOLATED_FREQUENCIES`: aequidistante positive Frequenzskala
- `INTERPOLATED_FFABS`: interpolierter Betrag `|F|`
- `NUM_INPUT_POINTS`: Eingangspunkte vor Filterung
- `NUM_FILTERED_INPUT_POINTS`: Punkte nach Filterung
- `MAX_INPUT_FREQUENCY`: letzte gueltige Messfrequenz nach Filterung
- `NUM_ITERATIONS`: tatsaechliche Iterationsanzahl
- `PEAK_CURRENT`, `FWHM`, `RMS_WIDTH`, `SKEWNESS`

## Code-Iststand vs. moegliche Stolperstellen

- `idx_extrap_start` wird von `extrapolate_low_and_high_frequencies()` zurueckgegeben,
  aber in `smooth_intermediate_ffsq()` nicht benutzt.
- Die Eingangstandardabweichung `INPUT_FFSQ_STD` wird bereinigt, aber im dokumentierten
  Rekonstruktionspfad nicht fuer die Modulus-Korrektur verwendet.
- Die Modulus-Korrektur nutzt stattdessen pauschal `20 %` des interpolierten `|F|`.
- Der DAQ-Array-Header verwendet feste Dimensionen `{2, 1024}`. Das passt zum Default,
  ist aber nicht dynamisch an `num_output_points_` gekoppelt.
- Die positive Frequenzskala enthaelt bei Default bis `2044 THz`, nicht exakt
  `2048 THz`, weil nur `N/2` Punkte mit Schritt `2 * max_frequency / N` erzeugt werden.

