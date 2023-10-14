import numpy as np
# from numba import njit

############################# SPECIFIC TO NB_PLAYERS #############################

NUMBER_PLAYERS = 2
NB_ROUNDS = 10

############################# TERRAIN CONSTANTS #############################

# Some terrains names have a leading 't' to differenciate with power
FORESTT  = 0
FARMLAND = 1
HILLT    = 2
SWAMPT   = 3
MOUNTAIN = 4
WATER    = 5

NOPOWERT = 0
CAVERN   = 1
MAGIC    = 2
MINE     = 3

############################# GAME CONSTANTS #############################

DICE_VALUES = [0, 0, 0, 1, 2, 3]
AVG_DICE = 1
MAX_DICE = 3

MAX_REDEPLOY = 8
DECK_SIZE = 6

IMMUNE_CONQUEST = 40
FULL_IMMUNITY   = 120

DECLINED_SPIRIT = 0
DECLINED = 1
ACTIVE   = 2

PHASE_READY          = 1 # Next action is to play
PHASE_CHOOSE         = 2 # Chose
PHASE_ABANDON        = 3 # Abandon
PHASE_CONQUEST       = 4 # Include preparation, attack, abandon, specialppl
PHASE_CONQ_WITH_DICE = 5 # Dice (not in berserk case)
PHASE_REDEPLOY       = 6 # Include redeploy, specialpower
PHASE_WAIT           = 7 # End of turn (after redeploy, or decline)

############################# PEOPLE CONSTANTS #############################

NOPPL     = 0
AMAZON    = 1  #  +4 pour attaque                                         DONE
DWARF     = 2  #  +1 victoire sur mine, même en déclin                    DONE
ELF       = 3  #  pas de défausse lors d'une défaite                      DONE
GHOUL     = 4  #  tous les zombies restent en déclin, peuvent attaquer    DONE
GIANT     = 5  #  -1 pour attaque voisin montagne                         DONE
HALFLING  = 6  #  départ n'importe où, immunité sur 2 prem régions        DONE
HUMAN     = 7  #  +1 victoire sur champs                                  DONE
ORC       = 8  #  +1 victoire pour région non-vide conquise               DONE
RATMAN    = 9  #  leur nombre                                             
SKELETON  = 10 #  +1 pion pour toutes 2 régions non-vide conquises        DONE
SORCERER  = 11 #  remplace pion unique adversaire actif par un sorcier    DONE
TRITON    = 12 #  -1 pour attaque région côtière                          DONE
TROLL     = 13 #  +1 défense sur chaque territoire même en déclin         DONE
WIZARD    = 14 #  +1 victoire sur source magique                          DONE
LOST_TRIBE= 15

MAX_SKELETONS = 20
MAX_SORCERERS = 18
#                       1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
initial_nb_people = [0, 6, 3, 6, 5, 6, 6, 5, 5, 8, 6, 5, 6, 5, 5, 2]
initial_tokens    = [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]


NOPOWER     = 0
ALCHEMIST   = 1  # +2 chaque tour      												DONE
BERSERK     = 2  # Lancer de dé AVANT chaque attaque                                DONE
BIVOUACKING = 3  # 5 défenses à placer à chaque tour + immunité au sorcier          DONE !
COMMANDO    = 4  # -1 attaque        												DONE !
DIPLOMAT    = 5  # Paix avec un peuple actif à choisir à chaque tour                DONE
DRAGONMASTER= 6  # 1 attaque dragon par tour + immunité complète                    DONE !
FLYING      = 7  # Toutes les régions sont voisines                                 DONE !
FOREST      = 8  # +1 victoire si forêt                                             DONE
FORTIFIED   = 9  # +1 défense avec forteresse mm en déclin, +1 par tour actif (max 6) DONE - doit limiter à +une fortress / tour
HEROIC      = 10 # 2 immunités complètes                                            DONE
HILL        = 11 # +1 victoire par colline                                          DONE
MERCHANT    = 12 # +1 victoire par région                                           DONE !
MOUNTED     = 13 # -1 attaque colline/ferme                                         DONE
PILLAGING   = 14 # +1 par région non vide conquise                                  DONE !
SEAFARING   = 15 # Conquête possible des mers/lacs, conservées en déclin            DONE
SPIRIT      = 16 # 2e peuple en déclin, et le reste jusqu'au bout                   DONE
STOUT       = 17 # Déclin possible juste après tour classique                       DONE
SWAMP       = 18 # +1 victoire par marais                                           DONE
UNDERWORLD  = 19 # -1 attaque caverne, et les cavernes sont adjacentes              DONE
WEALTHY     = 20 # +7 victoire à la fin premier tour								DONE !
#                        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
initial_nb_power   = [0, 4, 4, 5, 4, 5, 5, 5, 4, 3, 5, 4, 2, 5, 5, 5, 5, 4, 4, 5, 4]
initial_tokens_pwr = [0, 0, 0, 5, 0, 0, 0, 0, 0, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7]


