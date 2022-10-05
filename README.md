# Automated-PdM
Automated Prédictive Maintenance Systeme - Projet Transverse - M2 BDML Efrei Paris

<div style="text-align: right"> Mathys Goncalves - Nour-Eddine Oubenami - Julie Ngan - Lucie Bottin - Celine Khauv - Neal Louokdom</div>
</br>

## Le besoin

Aujourd'hui il existe deux types de maintenance qui sont utilisés par les entreprises : la maintenance corrective (ou curative) qui consiste à remplacer la pièce défaillante dans le système lorsque la panne survient et la maintenance préventive qui consiste à remplacer la pièce régulièrement sur une base de temps (1 an par exemple, la plupart du temps en suivant les indications constructeur)
La première utilise la pièce à 100% mais engendre des couts importants dus à l'indisponibilité de la machine. La seconde permet d'éviter le cout de l'indisponibilité importante mais engendre des maintenances plus régulières et sans utiliser 100% de la durée de vie de la pièce.

Il existe aujourd'hui une solution qui se développe dans le cadre de l'industrie 4.0 qui permet de consolider cette approche, on parle de maintenance prédictive.

Les enjeux financiers sont évidemment énormes. En moyenne, les outils industriels sont immobilisés 27 heures par mois. 

Selon Senseye, les coûts en euros d'une heure d'arrêt des machines en 2021 sont en moyenne les suivants : 

- Industrie automobile : 1 127 700 €
- Industrie pétrole et gaz : 184 700 €
- Industrie lourde : 157 410 €
- Industrie fabricants de Produits de Grande Consommation : 19 800€
L'impact financier de la maintenance prédictive dans les coûts d'exploitation peut donc être énorme.

Les sources d'économies pour adresser ces enjeux sont nombreuses : 

Meilleure fiabilité de l'outil de production et amélioration de la productivité
Diminution des pannes (temps d'arrêt machine) et amélioration de la disponibilité
Optimisation de la durée de vie des équipements industriels et optimisation de leur rentabilité
Optimisation de la gestion des pièces de rechange et baisse des coûts (pas de commande en dernière minute)
Réduction des coûts de maintenance globaux

Pour developper un systeme de maintenance prédictive il y a trois grands prérequis:
- Besoin de données en quantité et de qualité, sans quoi il n'y a pas de projet
- Besoin d'expertise metier, un opérateur qui connait bien ses machines 
- Besoin d'expertise de développement, un/des Data Scientist développant les solutions de maintenance predictive

La difficulté se situe autour des besoins d'expertise. En effet un data scientist sera necessaire pour le developpement des solutions mais n'aura aucune connaissance du domaine et aura besoin de l'expert metier qui lui est souvent indisponible. Ce genre de projets Data prennent souvent beaucoup de temps et la majorité n'aboutissent pas.

</br>

## Notre solution 
 
 Le projet s'inscrit dans la démarche de l'industry 4.0 vers laquelle toutes les industries s'orientent. 
 Grace aux données receuillit par l'IoT, l'objectif est de determiner les anomalies de fonctionnements des machines cela peut être compresseurs, pompes, tuyauteries ou tout autre machines equipés de capteurs. Nous souhaitons proposer une solution qui ne requiere aucune connaissances de developpement à l'utilisateur.

 Ici, nous ne proposons pas toutes les solutions d'analyses mais uniquement la mesure de la derive d'un equipement (ou piece d'un equipement) par rapport à son comportement normal. Cela est la solution qui peut être le plus automatisé mais il y a d'autres alternatives tel que : 
 - **Estimation de temps restant de vie (RUL)** : probleme supervisé de regression qui necessite une grande precision sur l'état de la machine, c'est un cas assez rare.
 - **Classsification de des instants avant la pannes (~48-96h)** : probleme supervisé de classification qui necessite une precision correct sur l'état de la machine, c'est un cas assez rare aussi bien que plus courant.
 - **Ciblage du probleme sur une valeur clée (valeur de temperature, de fuite de liquide etc.)** : probleme supervisé assez courant mais plus difficile dans l'automatisation. 

 Les paramètres de détection utilisés sont nombreux. Il peut s'agir :
- de vibration
- de température
- d'imagerie thermique
- de force exercée (jauge de contrainte ou déformation)
- de pression (déformation) 
- de couple pour des pièces en rotation
- de son

</br>

  ********

### Action utilisateur

Après s'etre login : 

1) Selection entre deux option : entrainement d'un nouveau model ou Prediciton en utilisant un model deja entrainé
2) Connexion d'un jeu de données 
3) Ajout d'indication supplementaire pour l'entrainement :
- Colonnes de date
- Selection de periodes à eliminer s'il y en a
- Indication de si on a un dataset de types long (plusieurs machines sur le dataset avec un colonne indiquant pour chaque ligne la machines )

### Action automatisé

1) Clean data
- Remove duplicates
- Drop null values/ populate
- Converting data types 

2) Filter out outliers 

3) Autoencoders modelling

4) Save Model

5) Deploy API

6) Notification (Telegram ?)

### Techno utilisées
