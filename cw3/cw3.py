from two_player_games.games.connect_four import ConnectFour  # or any other game
import random
import matplotlib.pyplot as plt


def score(current_player, state):
    """
    Evaluates the game state from the perspective of the current player.
    Positive scores favor the current player, and negative scores favor the opponent.
    """
    current_player_char = current_player.char
    opponent_char = '2' if current_player_char == '1' else '1'  # Determine opponent's char

    def count_sequences(player_char, length):
        """
        Counts the number of sequences of a given length for the specified player.
        """
        count = 0
        for sequence in state.get_all_lines():  # Get all rows, columns, and diagonals
            for i in range(len(sequence) - length + 1):
                window = sequence[i:i + length]
                if window.count(player_char) == length and window.count('.') == 4 - length:
                    count += 1
        return count

    # Assign weights to sequences of different lengths
    score = 0
    score += 1000 * count_sequences(current_player_char, 4)  # Winning move
    score += 10 * count_sequences(current_player_char, 3)    # Three in a row
    score += 1 * count_sequences(current_player_char, 2)     # Two in a row

    score -= 1000 * count_sequences(opponent_char, 4)        # Opponent's winning move
    score -= 10 * count_sequences(opponent_char, 3)          # Opponent's three in a row
    score -= 1 * count_sequences(opponent_char, 2)           # Opponent's two in a row

    return score


def minimax(state, depth, alpha, beta, maximizing_player):
    """
    Implements the minimax algorithm with alpha-beta pruning.
    For moves with the same score, it selects one randomly.
    """
    if depth == 0 or state.is_finished():
        # Evaluate the score of the current state
        current_player = state.get_current_player()
        return score(current_player, state)

    if maximizing_player:
        max_eval = float('-inf')
        best_moves = []
        for move in state.get_moves():
            # Simulate the move
            new_state = state.make_move(move)
            eval = minimax(new_state, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_moves = [move]  # Reset the list with the new best move
            elif eval == max_eval:
                best_moves.append(move)  # Add to the list of equally good moves
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval  # Always return the evaluation score
    else:
        min_eval = float('inf')
        best_moves = []
        for move in state.get_moves():
            # Simulate the move
            new_state = state.make_move(move)
            eval = minimax(new_state, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_moves = [move]  # Reset the list with the new best move
            elif eval == min_eval:
                best_moves.append(move)  # Add to the list of equally good moves
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval  # Always return the evaluation score


def simulate_games(depth_player1, depth_player2, num_games=10):
    """
    Simulates games between two AI players with different minimax depths.
    """
    results = {"Player 1 Wins": 0, "Player 2 Wins": 0, "Draws": 0}

    for game_number in range(1, num_games + 1):
        print(f"Simulating game {game_number}/{num_games} (Player 1 Depth: {depth_player1}, Player 2 Depth: {depth_player2})")
        game = ConnectFour()

        while not game.is_finished():
            moves = game.get_moves()

            if game.get_current_player().char == '1':  # Player 1 (AI) with depth depth_player1
                best_score = float('-inf')
                best_moves = []
                for move in moves:
                    new_state = game.state.make_move(move)
                    move_score = minimax(new_state, depth=depth_player1, alpha=float('-inf'), beta=float('inf'), maximizing_player=False)
                    if move_score > best_score:
                        best_score = move_score
                        best_moves = [move]  # Reset the list with the new best move
                    elif move_score == best_score:
                        best_moves.append(move)  # Add to the list of equally good moves
                best_move = random.choice(best_moves)  # Choose randomly among the best moves
                game.make_move(best_move)
            else:  # Player 2 (AI) with depth depth_player2
                best_score = float('-inf')
                best_moves = []
                for move in moves:
                    new_state = game.state.make_move(move)
                    move_score = minimax(new_state, depth=depth_player2, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
                    if move_score > best_score:
                        best_score = move_score
                        best_moves = [move]  # Reset the list with the new best move
                    elif move_score == best_score:
                        best_moves.append(move)  # Add to the list of equally good moves
                best_move = random.choice(best_moves)  # Choose randomly among the best moves
                game.make_move(best_move)

        winner = game.get_winner()
        if winner is None:
            results["Draws"] += 1
        elif winner.char == '1':
            results["Player 1 Wins"] += 1
        else:
            results["Player 2 Wins"] += 1

    return results


# Game simulation
depths = [2, 3, 4, 5]  # Player depths
num_games = 10
results_table = []

for depth_player1 in depths:
    row = []
    for depth_player2 in depths:
        if depth_player1 == depth_player2:
            row.append("N/A")  # Do not compare players with the same depth
        else:
            results = simulate_games(depth_player1, depth_player2, num_games)
            row.append(f"P1: {results['Player 1 Wins']}, P2: {results['Player 2 Wins']}, D: {results['Draws']}")
    results_table.append(row)

# Creating a table in matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

# Table headers
columns = [f"Player 2 Depth {d}" for d in depths]  # Columns represent Player 2 depths
rows = [f"Player 1 Depth {d}" for d in depths]    # Rows represent Player 1 depths

# Adding the table to the plot
table = ax.table(cellText=results_table, colLabels=columns, rowLabels=rows, loc='center', cellLoc='center')

# Table styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(columns))))

# Display the table
plt.title("Simulation Results: Player 1 (Rows) vs Player 2 (Columns)")
plt.show()