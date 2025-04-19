from two_player_games.games.connect_four import ConnectFour  # or any other game
import random
import matplotlib.pyplot as plt


def score2(current_player, state):
    """
    Evaluates the game state for the Connect Four game.
    Positive scores favor the current player, and negative scores favor the opponent.
    """
    current_player_char = current_player.char
    opponent_char = '2' if current_player_char == '1' else '1'  # Determine opponent's character

    def count_sequencesX(player_char, length):
        """
        Counts sequences of a given length for the specified player.
        Assigns scores based on the sequence length and the number of empty spaces around it.
        """
        score = 0
        for line in state.get_all_lines():  # Iterate over all rows, columns, and diagonals
            line_len = len(line)
            for i in range(line_len - length + 1):  # Sliding window of size `length`
                segment = line[i:i + length]
                if all(c == player_char for c in segment):  # Check if the segment contains only the player's tokens
                    # Determine empty spaces before and after the sequence
                    before = line[i - 1] if i - 1 >= 0 else None
                    after = line[i + length] if i + length < line_len else None

                    # Count consecutive empty spaces before and after
                    empty_before = 0
                    idx = i - 1
                    while idx >= 0 and line[idx] == '.':
                        empty_before += 1
                        idx -= 1

                    empty_after = 0
                    idx = i + length
                    while idx < line_len and line[idx] == '.':
                        empty_after += 1
                        idx += 1

                    # Assign scores based on sequence length and surrounding empty spaces
                    if length == 4:  # Winning sequence
                        score += 1000000
                    elif length == 3:  # Three in a row
                        if empty_before >= 1 and empty_after >= 1:
                            score += 12000  # Open-ended
                        elif empty_before >= 1 or empty_after >= 1:
                            score += 10000  # Half-open
                    elif length == 2:  # Two in a row
                        if empty_before >= 2 and empty_after >= 2:
                            score += 150  # Open-ended
                        elif (empty_before == 1 and empty_after >= 2) or (empty_before >= 2 and empty_after == 1):
                            score += 130  # Half-open with one side extended
                        elif (empty_before >= 1 and empty_after >= 1) or empty_before >= 2 or empty_after >= 2:
                            score += 100  # Half-open
        return score

    score = 0
    score += count_sequencesX(current_player_char, 4)  # Winning move
    score += count_sequencesX(current_player_char, 3)  # Three in a row
    score += count_sequencesX(current_player_char, 2)  # Two in a row

    score -= 2 * count_sequencesX(opponent_char, 4)  # Opponent's winning move
    score -= count_sequencesX(opponent_char, 3)  # Opponent's three in a row
    score -= count_sequencesX(opponent_char, 2)  # Opponent's two in a row

    return score


def minimax(state, depth, alpha, beta, maximizing_player):
    """
    Implements the minimax algorithm with alpha-beta pruning.
    Returns the best move and its evaluation score.
    If multiple moves have the same score, one is chosen randomly.

    Parameters:
    - state: The current game state.
    - depth: The maximum depth to search in the game tree.
    - alpha: The best score that the maximizing player can guarantee.
    - beta: The best score that the minimizing player can guarantee.
    - maximizing_player: True if the current player is maximizing, False otherwise.

    Returns:
    - best_move: The move that leads to the best evaluation score.
    - evaluation_score: The evaluation score of the best move.
    """
    if depth == 0 or state.is_finished():
        # Terminal state or maximum depth reached
        current_player = state.get_current_player()
        return None, score2(current_player, state)  # Evaluate the state

    if maximizing_player:
        max_eval = float('-inf')
        best_moves = []
        for move in state.get_moves():
            # Simulate the move
            new_state = state.make_move(move)
            _, eval = minimax(new_state, depth - 1, alpha, beta, not maximizing_player)
            if eval > max_eval:
                max_eval = eval
                best_moves = [move]  # Reset the list with the new best move
            elif eval == max_eval:
                best_moves.append(move)  # Add to the list of equally good moves
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        best_move = random.choice(best_moves)  # Randomly choose among the best moves
        return best_move, max_eval
    else:
        min_eval = float('inf')
        best_moves = []
        for move in state.get_moves():
            # Simulate the move
            new_state = state.make_move(move)
            _, eval = minimax(new_state, depth - 1, alpha, beta, not maximizing_player)
            if eval < min_eval:
                min_eval = eval
                best_moves = [move]  # Reset the list with the new best move
            elif eval == min_eval:
                best_moves.append(move)  # Add to the list of equally good moves
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        best_move = random.choice(best_moves)  # Randomly choose among the best moves
        return best_move, min_eval


def simulate_games(depth_player1, depth_player2, num_games=10, output_file="game_states.txt"):
    """
    Simulates games between two AI players with different minimax depths and writes game states to a file.

    Parameters:
    - depth_player1: The search depth for Player 1.
    - depth_player2: The search depth for Player 2.
    - num_games: The number of games to simulate.
    - output_file: The file to write the game states and results.

    Returns:
    - results: A dictionary containing the number of wins for each player and the number of draws.
    """
    results = {"Player 1 Wins": 0, "Player 2 Wins": 0, "Draws": 0}

    with open(str(depth_player1) + str(depth_player2) + output_file, "w") as file:  # Open the file for writing
        for game_number in range(1, num_games + 1):
            file.write(f"Simulating game {game_number}/{num_games} (Player 1 Depth: {depth_player1}, Player 2 Depth: {depth_player2})\n")
            print(f"Simulating game {game_number}/{num_games} (Player 1 Depth: {depth_player1}, Player 2 Depth: {depth_player2})\n")
            game = ConnectFour()

            while not game.is_finished():
                if game.get_current_player().char == '1':  # Player 1 (AI) with depth depth_player1
                    best_move, _ = minimax(game.state, depth=depth_player1, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
                    game.make_move(best_move)
                    file.write(f"Player 1 Move (Score: {_}):\n")
                    file.write(str(game.state) + "\n\n")  # Write the game state to the file
                else:  # Player 2 (AI) with depth depth_player2
                    best_move, _ = minimax(game.state, depth=depth_player2, alpha=float('-inf'), beta=float('inf'), maximizing_player=False)
                    game.make_move(best_move)
                    file.write(f"Player 2 Move (Score: {_}):\n")
                    file.write(str(game.state) + "\n\n")  # Write the game state to the file

            winner = game.get_winner()
            if winner is None:
                results["Draws"] += 1
                file.write("Result: Draw\n\n")
            elif winner.char == '1':
                results["Player 1 Wins"] += 1
                file.write("Result: Player 1 Wins\n\n")
            else:
                results["Player 2 Wins"] += 1
                file.write("Result: Player 2 Wins\n\n")

    return results


# Game simulation
depths = [1, 2, 3, 4, 5]  # Player depths
num_games = 10
results_table = []

for depth_player1 in depths:
    row = []
    for depth_player2 in depths:
        if depth_player1 == depth_player2 or depth_player1 < depth_player2:
            row.append("N/A")  # Do not compare players with the same depth or if Player 1 is weaker
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