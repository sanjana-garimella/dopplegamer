"""Explicit game-rule summaries and edge-case checklists for dashboard games."""

from __future__ import annotations

GAME_RULE_SPECS = {
    "Chess": {
        "rules": [
            "Standard FIDE chess with White to move first and orthodox initial setup.",
            "Legal move generation, check, checkmate, stalemate, repetition, the fifty-move rule, and insufficient material are delegated to python-chess.",
            "Pawn promotion defaults to queen in the UI when more than one promotion is legal.",
        ],
        "edge_cases": [
            "Illegal moves are rejected before they reach the board state.",
            "Draw outcomes from stalemate, repetition, fifty-move rule, and insufficient material are preserved.",
            "Game ends immediately on checkmate or draw claim conditions recognized by python-chess.",
        ],
    },
    "Othello": {
        "rules": [
            "Standard 8x8 Reversi opening position with Black moving first.",
            "A move is legal only if it brackets at least one contiguous line of opponent discs.",
            "If a player has no legal move they must pass; the game ends only when both players have no legal moves.",
            "Winner is the player with more discs when the game ends.",
        ],
        "edge_cases": [
            "Occupied squares and non-flipping moves are illegal.",
            "Single-pass turns are handled without ending the game early.",
            "Terminal state is triggered correctly on full board or back-to-back passes.",
        ],
    },
    "Checkers": {
        "rules": [
            "American checkers on dark squares only with 8x8 initial setup.",
            "Men move diagonally forward one square and capture diagonally forward by jumping a single adjacent opponent piece.",
            "Kings move and capture diagonally in both directions.",
            "Captures are mandatory, and a jumping piece must continue multi-jumps while legal.",
            "A man crowned on the back rank ends its move immediately in the current turn.",
        ],
        "edge_cases": [
            "Non-capturing moves are illegal when any capture exists.",
            "Player loses when starting a turn with no legal moves, even if pieces remain.",
            "Mid-jump state only exposes continuation moves for the jumping piece.",
        ],
    },
    "Connect Four": {
        "rules": [
            "Standard 7-column by 6-row gravity board.",
            "Pieces fall to the lowest open slot in a chosen column.",
            "Four connected horizontally, vertically, or diagonally wins immediately.",
        ],
        "edge_cases": [
            "Full columns are illegal.",
            "A full board without four-in-a-row is a draw.",
            "Winning move ends the game before the opponent acts.",
        ],
    },
    "Tic-Tac-Toe": {
        "rules": [
            "Standard 3x3 board with X moving first.",
            "Three in a row horizontally, vertically, or diagonally wins.",
            "A full board without a line is a draw.",
        ],
        "edge_cases": [
            "Occupied squares are illegal.",
            "Winning move ends the game immediately.",
            "No extra move is applied after the final board-filling move.",
        ],
    },
    "RPS+": {
        "rules": [
            "Base moves are Rock, Paper, Scissors, and Lizard.",
            "Power costs 2 energy and defeats any base move.",
            "Recharge restores 1 energy up to a cap of 5 but loses to any non-Recharge move.",
            "Players begin with 3 energy and Power is legal only at 2 or more energy.",
        ],
        "edge_cases": [
            "Illegal Power attempts are blocked by legal-move generation.",
            "Energy is capped and never drops below zero.",
            "Ties preserve score while still applying Recharge and Power energy updates.",
        ],
    },
    "2048": {
        "rules": [
            "Original 4x4 slide-and-merge rules with one merge per tile per move.",
            "After each successful move, exactly one new tile spawns in an empty cell.",
            "Spawn distribution is 90% for tile 2 and 10% for tile 4.",
        ],
        "edge_cases": [
            "Moves that do not change the board are illegal.",
            "Chain merges happen from the movement direction only once per tile.",
            "Win state is tracked when a 2048 tile appears; loss occurs when no legal moves remain.",
        ],
    },
    "Wordle": {
        "rules": [
            "Five-letter guesses with six total attempts.",
            "Green letters match exact position; yellow letters exist elsewhere in the target word; gray letters do not count against unmatched remaining target letters.",
            "Duplicate-letter scoring follows the original remaining-letter accounting behavior.",
        ],
        "edge_cases": [
            "Repeated guesses are allowed.",
            "Duplicate letters are only marked yellow as many times as the target supports after greens are assigned.",
            "Game ends immediately on a correct guess or after the sixth guess.",
        ],
    },
    "Sudoku": {
        "rules": [
            "Standard 9x9 Sudoku with digits 1-9.",
            "No duplicate digit may appear in a row, column, or 3x3 box.",
            "Only originally empty cells are editable.",
        ],
        "edge_cases": [
            "Prefilled cells cannot be changed.",
            "Illegal placements are rejected rather than auto-corrected.",
            "Solved state is only valid when the full board matches the solution.",
        ],
    },
    "Pac-Man": {
        "rules": [
            "High-fidelity approximation of arcade Pac-Man using grid movement, pellets, power pellets, wrap tunnels, and ghost pursuit.",
            "Pac-Man moves one tile per turn along legal corridors.",
            "Eating a power pellet activates frightened mode for a limited number of turns.",
            "All pellets must be cleared to win the level; contact with an unfleeing ghost loses the game.",
        ],
        "edge_cases": [
            "Wall collisions are prevented.",
            "Tunnel wrapping preserves horizontal travel across side exits.",
            "Ghost collisions are evaluated after both movement phases and frightened ghosts can be eaten instead of causing a loss.",
        ],
    },
}


GAME_HINTS = {
    "Tic-Tac-Toe": [
        "The center square is usually the strongest opening move.",
        "Forks matter more than single threats once the board opens up.",
    ],
    "Connect Four": [
        "The center column touches the most winning lines.",
        "Threats that create two winning replies are usually the real goal.",
    ],
    "Chess": [
        "Develop pieces before chasing pawns.",
        "Checks, captures, and threats are the first moves worth scanning.",
    ],
    "Othello": [
        "Corners are stable and usually worth more than raw disc count.",
        "Avoid making your frontier discs too wide too early.",
    ],
    "Checkers": [
        "Forced captures can still be tactical traps.",
        "Preserve back-rank defense until you can king safely.",
    ],
    "Gomoku": [
        "Open threes and open fours decide most quick games.",
        "Blocking an opponent open four is mandatory.",
    ],
}


GAME_CHALLENGES = {
    "Chess": [
        {
            "id": "sicilian_pressure",
            "label": "Sicilian Pressure",
            "description": "Jump into a tactical Sicilian structure after the opening.",
        },
        {
            "id": "endgame_conversion",
            "label": "Endgame Conversion",
            "description": "Play from a sparse board where clean technique matters.",
        },
    ],
    "Checkers": [
        {
            "id": "forced_chain",
            "label": "Forced Jump Chain",
            "description": "Start in a capture sequence where the trap is already live.",
        }
    ],
    "Connect Four": [
        {
            "id": "center_tension",
            "label": "Center Tension",
            "description": "Begin from a balanced midgame with contested center columns.",
        }
    ],
    "Gomoku": [
        {
            "id": "open_three_race",
            "label": "Open Three Race",
            "description": "Resume from a live threat-building position in the midgame.",
        }
    ],
}


CLONE_LADDER_RPS = [
    {
        "agent": "random",
        "title": "Calibration",
        "description": "Warm-up rung against pure randomness. Use it to establish your own baseline tempo.",
    },
    {
        "agent": "heuristic",
        "title": "Pattern Pressure",
        "description": "The bot starts reacting to obvious habits and tempo leaks.",
    },
    {
        "agent": "ngram",
        "title": "Short-Memory Clone",
        "description": "A context-based clone that imitates your recent move sequencing.",
    },
    {
        "agent": "lstm",
        "title": "Deep Sequence Clone",
        "description": "A sequence model that tracks longer behavioral rhythms and streaks.",
    },
    {
        "agent": "adaptive_router",
        "title": "Mirror Boss",
        "description": "A routing opponent that blends clone pressure with competitive adaptation.",
    },
]


CHALLENGE_PACKS = {
    "Chess": {
        "Opening Traps": ["sicilian_pressure"],
        "Endgame Conversion": ["endgame_conversion"],
    },
    "Checkers": {
        "Bias Pressure": ["forced_chain"],
    },
    "Connect Four": {
        "Opening Traps": ["center_tension"],
    },
    "Gomoku": {
        "Bias Pressure": ["open_three_race"],
    },
}
