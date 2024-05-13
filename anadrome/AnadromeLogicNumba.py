import numba
import numpy as np
from numba import njit

############################## BOARD DESCRIPTION ##############################

# Obstacle cards are represented using 1 line
#####   0      1      2      3
##### Predator Rapid  Dam  #Trout on it

# Fish cards are represented using 1 line
#####                    0        1        2       3
##### card state:       Owner     Position Usages Trout
# Where
## Owner = Player owning the card (1-4, -1 if it's on the river, 0 if it's in the deck or -2 if it's in the discard pile)
## Position = Either on the river (0-12), in the player area (0-...), or 0 for the deck or discard pile
## Usages = Usages left (2/1/0), 2=in hand, 1=on table, 0=with trout
## Trout = Contains trout of player x (1/2/3/4), -1=neutral trout, 0=no trout

# Ability cards:
#####  0                    1                     2                   3
#####  In the game (1/0)   Chosen by player X   First player marker   -

# Player lines contain:
#####  0                1                   2                3
#####  #personal trout  #neutral trout      Salmon pos.   Passed?

# First line describes what is needed to buy the card, second line describes
# Here is the description of each line of the board. For readibility, we defined
# "shortcuts" that actually are views (numpy name) of overal board.
##### Index (2p)  Index (4p)   Shortcut          	Meaning
#####   0         0         self.game_info        Game info: Week number, round number, starting player number
#####   1-12      1-12      self.obstacles    The randomized obstacle cards
#####   13-63     13-63     self.cards        All fish cards
#####   64-71     64-79     self.starting     All starting cards
#####   72        80        self.abilities    The ability cards
#####   72        80        self.players      The state per player

############################## ACTION DESCRIPTION #############################
# There are ... actions. Here is description of each action:
##### Index    Meaning
#####   0      Buy fish card in position 0
#####  ...
#####  12      Buy fish card in position 12
#####  13-63   Do action of card X by playing it from your hand
#####  64-114  Do action of card X by activating it using a personal trout
#####  115-165 Do action of card X by activating it using a neutral trout
#####  166-216 Gain resource of card X
#####  217-267 Pay by discarding card X
#####  268     Pay by discarding a trout
#####  269     Advance earshot
#####  270     Pass

RIVER_LENGTH = 13
NB_CARDS = 51
NB_STARTING_CARDS = 4


@njit(cache=True, fastmath=True, nogil=True)
def observation_size(num_players):
	return RIVER_LENGTH + NB_CARDS + NB_STARTING_CARDS * num_players + 1, 4


@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return RIVER_LENGTH + 5 * NB_CARDS + 3


spec = [
	('num_players', numba.int8),
	('current_player_index', numba.int8),

	('game_info', numba.int8[:, :]),
	('obstacles', numba.int8[:, :]),
	('fish_cards', numba.int8[:, :]),
	('starting_cards', numba.int8[:, :]),
	('players_state', numba.int8[:, :]),
]


@numba.experimental.jitclass(spec)
class Board:
	def __init__(self, num_players):
		n = num_players
		self.num_players = n
		self.current_player_index = 0
		self.max_moves = 64 * num_players
		self.score_win = 13
		self.state = np.zeros(observation_size(self.num_players), dtype=np.int8)
		self.init_game()

	def init_game(self):
		# Random number generator
		prng = np.random.default_rng()

		self.copy_state(np.zeros(observation_size(), dtype=np.int8), copy_or_not=False)

		# Set up the river with obstacles
		for pos in range(0, 3):
			self.obstacles[pos] = [1, 0, 0, 0]
		for pos in range(4, 7):
			self.obstacles[pos] = [0, 1, 0, 0]
		for pos in range(8, 11):
			self.obstacles[pos] = [0, 0, 1, 0]
		prng.shuffle(self.obstacles)

		# Set up the cards on the river
		fish_card_indexes = prng.choice(NB_CARDS, size=RIVER_LENGTH, replace=False)
		for i, index in enumerate(fish_card_indexes):
			self.fish_cards[i,] = fish_card_indexes[index]

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state
		n = self.num_players
		self.game_info = self.state[0, :]  # 1      # Game state info
		self.obstacles = self.state[1: 12, :]  # 12      # Obstacle cards
		self.fish_cards = self.state[12:12 + NB_CARDS, :]  # 51      # All fish cards
		# n*4   # Numbers of starting cards for each player
		idx_post_st = 12 + NB_CARDS + (NB_STARTING_CARDS * n)
		self.starting_cards = self.state[12 + NB_CARDS:idx_post_st, :]  # .reshape(4, n)
		# n*1    # Trout, ability, salmon pos., hasPassed, per player
		self.players_trout = self.state[idx_post_st:idx_post_st + n, 0]
		self.players_ability = self.state[idx_post_st:idx_post_st + n, 1]
		self.players_pos = self.state[idx_post_st:idx_post_st + n, 2]
		self.players_passed = self.state[idx_post_st:idx_post_st + n, 3]
