# Documentation — Dashboard Efficacité des Serveurs
## Projet : The Kitchen

---

## 1. Contexte et Objectif

The Kitchen est un restaurant équipé de **6 caméras de surveillance IMOU** qui filment la salle en permanence. Le système d'intelligence artificielle analyse ces vidéos en temps réel pour suivre chaque serveur automatiquement.

Le but du dashboard est de donner au **manager** une vue claire et en temps réel sur **l'efficacité de chaque serveur** pendant son service, sans aucune intervention manuelle. Le système reconnaît chaque serveur par son visage et son uniforme, puis mesure son comportement tout au long du service.

---

## 2. Ce que le Dashboard doit afficher

### 2.1 En-tête général (vue globale du restaurant)

Le haut du dashboard affiche la situation globale du restaurant à cet instant :

- **Nombre de serveurs actifs** en ce moment dans la salle
- **Nombre de clients** présents dans le restaurant
- **Nombre de tables occupées** sur le total des tables disponibles
- **Temps d'attente moyen** : combien de temps un client attend en moyenne avant qu'un serveur vienne le voir
- **Heure de début du service** en cours

---

### 2.2 Carte individuelle pour chaque serveur

Pour chaque serveur reconnu, le dashboard affiche une carte personnelle avec :

#### Identité
- **Nom du serveur** (identifié automatiquement par reconnaissance faciale)
- **Caméra** sur laquelle il a été vu en dernier
- **Heure d'arrivée** dans la salle et **durée totale de service**

#### Score d'efficacité global
- Un **score de 0 à 100** affiché sous forme de jauge circulaire colorée :
  - **Vert** (70 à 100) : serveur très efficace
  - **Orange** (40 à 69) : efficacité moyenne, à surveiller
  - **Rouge** (0 à 39) : serveur peu actif, alerte manager

#### Métriques détaillées (voir Section 3)
- Vitesse de déplacement
- Tables visitées
- Temps debout
- Réactivité avec les clients

#### Alertes
- 🔴 **Alerte inactivité** : le serveur n'a pas bougé depuis plus d'1 minute
- 🟡 **Alerte table non servie** : une table occupée n'a pas été visitée depuis plus de 10 minutes

---

### 2.3 Carte thermique de la salle

Une vue du plan du restaurant avec :

- La **position en temps réel** de chaque serveur (point coloré par nom)
- Les **zones où chaque serveur passe le plus de temps** (carte de chaleur colorée)
- Le **statut de chaque table** : libre, occupée, visitée, en attente

Cela permet au manager de voir d'un coup d'œil si un serveur couvre bien sa zone ou s'il reste toujours au même endroit.

---

### 2.4 Graphiques historiques

En bas du dashboard, des graphiques montrant l'évolution pendant le service :

- **Courbe du score d'efficacité** de chaque serveur au fil du temps
- **Histogramme des temps de service** : comparaison des serveurs
- **Activité par tranche de 15 minutes** : quand est-ce que chaque serveur était le plus actif
- **Nombre de tables servies par heure** pour chaque serveur

---

## 3. Les Métriques qui Déterminent l'Efficacité

### 3.1 Formule du Score d'Efficacité Global

Le score final affiché pour chaque serveur est calculé comme suit :

```
Score d'efficacité (0-100) =
    (Vitesse de déplacement  × 30%)
  + (Réactivité aux clients  × 30%)
  + (Couverture des tables   × 25%)
  + (Temps actif debout      × 15%)
```

---

### 3.2 Détail de chaque métrique

---

#### Métrique 1 — Vitesse de Déplacement (poids : 30%)

**Ce que ça mesure :** À quelle vitesse le serveur se déplace dans la salle.

**Comment c'est calculé :** Le système suit la position du serveur frame par frame et mesure la distance parcourue. On obtient une vitesse moyenne en pixels par seconde, qu'on convertit en "rapide/normal/lent".

**Interprétation :**
- Un serveur rapide et actif obtient un score élevé dans cette catégorie
- Un serveur qui reste immobile pendant de longues périodes obtient un score bas
- Seuil d'alerte : moins de 1 pixel/frame pendant plus de 60 secondes = inactivité détectée

**Indicateur visuel sur le dashboard :** Barre de progression + mini-graphique d'évolution

---

#### Métrique 2 — Réactivité aux Clients (poids : 30%)

**Ce que ça mesure :** Combien de temps s'écoule entre le moment où un client s'assoit à une table et le moment où le serveur vient le voir pour la première fois.

**Comment c'est calculé :** Le système détecte qu'une table vient d'être occupée (client assis détecté), puis chronomètre jusqu'à la première visite d'un serveur (serveur détecté à proximité de cette table en position debout).

**Interprétation :**
- Moins de 2 minutes → Excellent
- 2 à 5 minutes → Bon
- 5 à 10 minutes → Passable
- Plus de 10 minutes → Insuffisant, alerte déclenchée

**Indicateur visuel sur le dashboard :** Chronomètre par table / Temps moyen affiché par serveur

---

#### Métrique 3 — Couverture des Tables (poids : 25%)

**Ce que ça mesure :** Combien de tables différentes le serveur a visitées pendant son service, par rapport au nombre total de tables dans le restaurant.

**Comment c'est calculé :** À chaque fois qu'un serveur s'approche d'une table (son centre est dans la zone de la table) et qu'il est en position debout, la table est comptée comme "visitée" par ce serveur. On calcule le ratio : tables visitées / total tables.

**Interprétation :**
- Un serveur qui visite toutes les tables obtient un score maximal
- Un serveur qui reste concentré sur 1 ou 2 tables a un score bas
- Cela permet de détecter si un serveur a une zone bien définie ou s'il est polyvalent

**Indicateur visuel sur le dashboard :** Représentation graphique des tables avec celles visitées colorées

---

#### Métrique 4 — Temps Actif Debout (poids : 15%)

**Ce que ça mesure :** Le pourcentage du temps total où le serveur est en position debout et active, par opposition à être assis ou immobile.

**Comment c'est calculé :** Le système analyse la forme de la silhouette détectée. Si la hauteur est nettement supérieure à la largeur (ratio h/w > 1.8), la personne est considérée debout. On calcule le pourcentage de frames où c'est le cas.

**Interprétation :**
- Un serveur debout et en mouvement est actif dans son service
- Cette métrique distingue un serveur qui fait son travail d'une personne assise (client ou serveur au repos)

**Indicateur visuel sur le dashboard :** Pourcentage affiché avec barre de progression

---

### 3.3 Métriques Complémentaires (informatives, non inclues dans le score)

Ces données sont affichées sur le dashboard à titre informatif, sans être incluses dans le calcul du score principal :

| Métrique | Description |
|---|---|
| **Durée totale de présence** | Depuis quand le serveur est dans la salle ce service |
| **Nombre total de tables visitées** | Compteur brut de passages à une table |
| **Score de reconnaissance** | Niveau de confiance de l'identification du serveur (visage + uniforme) |
| **Dernière position connue** | Dernière zone où le serveur a été vu |
| **Caméra principale** | Caméra où le serveur apparaît le plus souvent |

---

## 4. Alertes Automatiques pour le Manager

Le dashboard génère des alertes visuelles et sonores dans les cas suivants :

| Situation | Condition | Niveau |
|---|---|---|
| Serveur inactif | Vitesse < 1 px/frame pendant > 1 minute | 🔴 Critique |
| Table non servie | Client à table depuis > 10 min sans visite | 🔴 Critique |
| Réactivité faible | Temps moyen > 7 minutes | 🟡 Avertissement |
| Faible couverture | Moins de 30% des tables visitées | 🟡 Avertissement |
| Score bas | Efficacité globale < 40 pendant 30+ min | 🟡 Avertissement |

---

## 5. Comment le Système Identifie Chaque Serveur

Pour que les métriques soient correctement attribuées à chaque serveur (et non à un client), le système utilise deux méthodes combinées :

### Reconnaissance faciale (ArcFace)
Le système compare le visage détecté avec la base de données des photos de serveurs. Quand la ressemblance est suffisante (score > 0.45), le serveur est identifié par son nom. Cette méthode a la priorité.

### Reconnaissance par l'uniforme et le corps
Si le visage n'est pas visible (personne de dos, mauvaise lumière), le système utilise un modèle entraîné sur les uniformes "The Kitchen" pour identifier chaque serveur par son apparence générale. Cette méthode sert de secours.

### Règle de verrouillage
Une fois qu'un serveur est identifié avec suffisamment de confiance, son identité est **verrouillée** sur son identifiant de tracking. Cela évite que le nom change à chaque frame et garantit que les métriques s'accumulent correctement sur la bonne personne.

---

## 6. Résumé : Ce dont le Dashboard a Besoin

Pour que le dashboard fonctionne, le système de détection doit lui fournir, en temps réel, pour chaque serveur :

1. **Son nom** (identifié par ArcFace ou classificateur)
2. **Sa position actuelle** dans la salle (coordonnées x, y)
3. **Sa vitesse de déplacement** à cet instant
4. **Le nombre de tables qu'il a visitées** depuis le début du service
5. **Le temps moyen** qu'il met pour aller voir un client
6. **Son pourcentage de temps debout**
7. **La durée de sa présence** dans la salle
8. **Son score d'efficacité calculé** (0-100)
9. **Les alertes actives** le concernant

---

## 7. Prochaines Étapes pour Implémenter le Dashboard

Le système de détection est **déjà opérationnel**. Ce qui reste à faire pour avoir le dashboard complet :

1. **Ajouter une API** dans le script de détection pour envoyer les données en temps réel (FastAPI est déjà listé comme dépendance du projet)
2. **Créer l'interface visuelle** du dashboard (page web qui affiche les métriques reçues)
3. **Connecter l'interface** à l'API via WebSocket pour la mise à jour en temps réel
4. **Configurer les alertes** avec les seuils définis en Section 4
5. **Ajouter la sauvegarde historique** pour pouvoir générer des rapports journaliers

---

*Document rédigé le 19 février 2026 — The Kitchen AI System*
