import chess
import random
import time
from collections import defaultdict

transposition_table = defaultdict(dict)
opening_book = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": [
        "d7d5",  # Scandinavian Defense
        "g8f6",  # Nimzo-Indian
        "e7e5"   # Sicilian
    ],
    # Sicilian Defense (1.e4 c5)
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["d7d6", "g8f6"],
    # Italian Game (1.e4 e5 2.Nf3 Nc6)
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": ["f8c5"],
        # Caro-Kann (1.e4 c6)
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["d7d5"],
    
    # French Defense (1.e4 e6)
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["d7d5"],
    
    # Ruy Lopez (1.e4 e5 2.Nf3 Nc6 3.Bb5)
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3": ["a7a6"],
}

piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
} 

killer_moves = defaultdict(list)

piece_square_tables = {
    chess.PAWN: [
      [0, 0, 0, 0, 0, 0, 0, 0],
      [5, 5, 5, 5, 5, 5, 5, 5],
      [1, 1, 2, 3, 3, 2, 1, 1],
      [0, 0, 0, 2, 2, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [-5, -5, -5, -5, -5, -5, -5, -5],
      [0, 0, 0, 0, 0, 0, 0, 0]
    ],
    chess.KNIGHT: [
      [-5, -4, -3, -3, -3, -3, -4, -5],
      [-4, -2, 0, 0, 0, 0, -2, -4],
      [-3, 0, 1, 1, 1, 1, 0, -3],
      [-3, 0, 1, 2, 2, 1, 0, -3],
      [-3, 0, 1, 2, 2, 1, 0, -3],
      [-3, 0, 1, 1, 1, 1, 0, -3],
      [-4, -2, 0, 0, 0, 0, -2, -4],
      [-5, -4, -3, -3, -3, -3, -4, -5]
    ],
    chess.BISHOP: [
        [-2, -1, -1, -1, -1, -1, -1, -2],
        [-1, 0.5, 0, 0, 0, 0, 0.5, -1],
        [-1, 1, 1, 1, 1, 1, 1, -1],
        [-1, 0, 1, 1, 1, 1, 0, -1],
        [-1, 0.5, 0.5, 1, 1, 0.5, 0.5, -1],
        [-1, 0, 0.5, 1, 1, 0.5, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1],
        [-2, -1, -1, -1, -1, -1, -1, -2]
    ],
    chess.ROOK: [
        [0, 0, 0, 0.5, 0.5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 1, 1, 1, 1, 1, 1, 0.5],
        [0.5, 1, 1, 1, 1, 1, 1, 0.5],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.5, 0.5, 0, 0, 0]
    ],
    chess.QUEEN: [
        [-2, -1, -1, -0.5, -0.5, -1, -1, -2],
        [-1, 0, 0.5, 0, 0, 0, 0, -1],
        [-1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, -1],
        [-0.5, 0, 0.5, 1, 1, 0.5, 0, -0.5],
        [-0.5, 0, 0.5, 1, 1, 0.5, 0, -0.5],
        [-1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, -1],
        [-1, 0, 0, 0, 0, 0.5, 0, -1],
        [-2, -1, -1, -0.5, -0.5, -1, -1, -2]
    ],
    chess.KING: [
        [-3, -4, -4, -5, -5, -4, -4, -3],
        [-3, -4, -4, -5, -5, -4, -4, -3],
        [-3, -4, -4, -5, -5, -4, -4, -3],
        [-3, -4, -4, -5, -5, -4, -4, -3],
        [-2, -3, -3, -4, -4, -3, -3, -2],
        [-1, -2, -2, -2, -2, -2, -2, -1],
        [2, 2, 0, 0, 0, 0, 2, 2],
        [2, 3, 1, 0, 0, 1, 3, 2]
    ]
}

def get_pawn_in_file(board, file):
    return any(
        board.piece_at(chess.square(file, rank)) and
        board.piece_at(chess.square(file, rank)).piece_type == chess.PAWN
        for rank in range(8)
    )

def get_opening_move(board):
    """Returns a move from the opening book if available."""
    fen = board.fen().split(" ")[0]     # Position-only FEN
    return opening_book.get(fen, [])

def get_piece_value(piece, square):
    """Positional Value calculation"""
    table = piece_square_tables.get(piece.piece_type, [[0]*8 for _ in range(8)])
    row = 7 - (square // 8) if piece.color == chess.WHITE else square // 8
    col = square % 8
    return table[row][col] * 0.1

def evaluate_passed_pawns(board, color):
    """Reward passed pawns (pawns with no opposing pawns in front.)"""
    passed_pawn_bonus = 0
    our_pawns = board.pieces(chess.PAWN, color)
    their_pawns = board.pieces(chess.PAWN, not color)
    
    for pawn_sq in our_pawns:
        file = chess.square_file(pawn_sq)
        rank = chess.square_rank(pawn_sq)
        
        has_obstacle = False
        for f in [file-1, file, file+1]:
            if f < 0 or f > 7:
                continue
            if their_pawns & chess.BB_RANK_ATTACKS[pawn_sq][f]:
                has_obstacle = True
                break
            
        if not has_obstacle:
            bonus = rank if color == chess.WHITE else 7 - rank
            passed_pawn_bonus += bonus * 0.2
            
    return passed_pawn_bonus

def evaluate_pawn_structure(board, color):
    pawns = board.pieces(chess.PAWN, color)
    files = [chess.square_file(sq) for sq in pawns]
    
    doubled = sum(files.count(f) > 1 for f in set(files)) * 0.5
    isolated = sum(
        1 for f in files
        if (f == 0 or (f-1) not in files) and
           (f == 7 or (f+1) not in files)
    )
        
    return -(doubled + isolated)    # Penalty

def evaluate_king_safety(board, color):
    king_square = board.king(color)
    if not king_square:
        return 0
    
    # Penalize if king is in center during middlegame
    file = chess.square_file(king_square)
    rank = chess.square_rank(king_square)
    safety = 0
    
    if not is_endgame(board):  # Middlegame
        safety -= abs(3.5 - file) + abs(3.5 - rank)       
    return safety

def evaluate_bishop_pair(board, color):
    bishops = board.pieces(chess.BISHOP, color)
    if len(bishops) >= 2:
        return 0.5  # Bonus for having two bishops
    return 0

def evaluate_rook_files(board, color):
    rooks = board.pieces(chess.ROOK, color)
    # pawns = board.pieces(chess.PAWN, color) | board.pieces(chess.PAWN, not color)
    open_file_bonus = 0
    
    for rook_sq in rooks:
        file = chess.square_file(rook_sq)
        has_pawn = any(
            board.piece_at(chess.square(file, rank)) and
            board.piece_at(chess.square(file, rank)).piece_type == chess.PAWN
            for rank in range(8)
        )
        
        if not has_pawn:
            open_file_bonus += 0.7  # Rook on open file
        elif not board.pieces(chess.PAWN, color) & chess.BB_FILES[file]:
            open_file_bonus += 0.3  # Rook on semi-open file
            
    return open_file_bonus

def evaluate_promotion(board, color):
    """Reward advanced pawns."""
    if not is_endgame(board):
        return 0
    
    promotion_bonus = 0
    pawns = board.pieces(chess.PAWN, color)
    for pawn_sq in pawns:
        rank = chess.square_rank(pawn_sq) if color == chess.WHITE else 7 - chess.square_rank(pawn_sq)
        if rank >= 5:   # Pawn on 6th/7th rank (White) or 3rd/2nd (Black)
            promotion_bonus += (rank - 4) * 0.5     # +0.5 for 6th, +1.5 for 7th
    return promotion_bonus

def evaluate_center_control(board, color):
    """Reward pieces controlling central squares (d4/d5/e4/e5)."""
    center = chess.SquareSet([chess.D4, chess.D5, chess.E4, chess.E5])
    control = 0
    
    for square in center:
        attacker = board.color_at(square)
        if attacker == color:
            control += 0.5
        elif board.is_attacked_by(color, square):
            control += 0.3
            
    return control

def evaluate_threats(board, color):
    """Penalize undefended pieces and reward attacks on valuable targets."""
    threat_score = 0
    our_pieces = board.occupied_co[color]
    their_pieces = board.occupied_co[not color]
    
    for square in chess.SquareSet(our_pieces):
        piece_value = piece_values.get(board.piece_type_at(square), 0)
        attackers = board.attackers(not color, square)
        
        if attackers:
            defenders = board.attackers(color, square)
            threat_score -= (len(attackers) - len(defenders)) * piece_value * 0.2
            
    for square in chess.SquareSet(their_pieces):
        piece_value = piece_values.get(board.piece_type_at(square), 0)
        attackers = board.attackers(color, square)
        
        if attackers:
            threat_score += len(attackers) * piece_value * 0.1
            
    return threat_score

def evaluate_board(board):
    
    Score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            material = piece_values[piece.piece_type]
            positional = get_piece_value(piece, square)
            total = material + positional
            
            if piece.color == chess.BLACK:      # AI is Black
                Score += total
            else:
                Score -= total
                
    mobility = len(list(board.legal_moves))
    if board.turn == chess.BLACK:
        Score += mobility * 0.05
    else:
        Score -= mobility * 0.05
        
    Score += evaluate_pawn_structure(board, chess.BLACK)    # AI's Pawns
    Score -= evaluate_pawn_structure(board, chess.WHITE)    # User's Pawns
    
    Score += evaluate_king_safety(board, chess.BLACK)
    Score -= evaluate_king_safety(board, chess.WHITE)
    
    Score += evaluate_king_activity(board, chess.BLACK)
    Score -= evaluate_king_activity(board, chess.WHITE)
    
    Score += evaluate_promotion(board, chess.BLACK)
    Score -= evaluate_promotion(board, chess.WHITE)
    
    Score += evaluate_bishop_pair(board, chess.BLACK)
    Score -= evaluate_bishop_pair(board, chess.WHITE)
    
    Score += evaluate_rook_files(board, chess.BLACK)
    Score -= evaluate_rook_files(board, chess.WHITE)
    
    Score += evaluate_passed_pawns(board, chess.BLACK)
    Score -= evaluate_passed_pawns(board, chess.WHITE)
    
    Score += evaluate_center_control(board, chess.BLACK)
    Score -= evaluate_center_control(board, chess.WHITE)
    
    Score += evaluate_threats(board, chess.BLACK)
    Score -= evaluate_threats(board, chess.WHITE)
    
    return Score 

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return quiescence_search(board, alpha, beta)
    
    fen = board.fen()
    if fen in transposition_table:
        cached_depth, cached_eval = transposition_table[fen]
        if cached_depth >= depth:
            return cached_eval
        
    best_eval = -float('inf') if maximizing_player else float('inf')        
    for move in order_moves(board, board.legal_moves, depth):
        board.push(move)
        eval = minimax(board, depth-1, alpha, beta, not maximizing_player)
        board.pop()
        
        if eval >= beta and move not in killer_moves[depth]:
            killer_moves[depth] = [move] + killer_moves[depth][:1]
        
        if maximizing_player:
            best_eval = max(best_eval, eval)
            alpha = max(alpha, eval)
        else:
            best_eval = min(best_eval, eval)
            beta = min(beta, eval)
            
        if beta <= alpha:
            break
        
                
    transposition_table[fen] = (depth, best_eval)
    return best_eval

def quiescence_search(board, alpha, beta):
    stand_pat = evaluate_board(board)
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)
        
    # Only evaluate captures
    for move in board.generate_legal_captures():
        board.push(move)
        eval = -quiescence_search(board, -beta, -alpha)
        board.pop()
        
        if eval >= beta:
            return beta
        alpha = max(alpha, eval)
            
    return alpha

def order_moves(board, moves, depth=0):
    """Optimized move ordering without board manipulation"""
    killers = killer_moves.get(depth, [])
    return sorted(moves, key=lambda m: (
        m in killers,
        -int(board.is_capture(m)),
        -int(board.gives_check(m)),
        -piece_values.get(board.piece_type_at(m.to_square), 0)
    ), reverse= True)
    
def get_ai_move(board, max_depth= 3):
    """Improved iterative deepening with time management"""
    opening_moves = get_opening_move(board)
    if opening_moves:
        legal_opening_moves = [chess.Move.from_uci(uci) for uci in opening_moves]
        legal_opening_moves = [m for m in legal_opening_moves if m in board.legal_moves]
        if legal_opening_moves:
            return random.choice(legal_opening_moves)
        
    best_move = None
    start_time = time.time()

    for depth in range(1, max_depth + 1):
        current_best = None
        best_eval = -float('inf')
        
        if time.time() - start_time > 2:
            break
    
        for move in order_moves(board, board.legal_moves):
            board.push(move)
            eval = minimax(board, depth-1, -float('inf'), float('inf'), False)
            board.pop()
            
            if eval > best_eval:
                best_eval = eval
                current_best = move
                
        if current_best:
            best_move = current_best
    
    return best_move or random.choice(list(board.legal_moves))

def play_chess():
    board = chess.Board()
    
    while not board.is_game_over():
        print(board.unicode(borders=True))
        
        if board.turn == chess.WHITE:
            # User move
            try:
                board.push_uci(input("Your move (e.g. 'e2e4': "))
            except:
                print("Invalid move! Try again.")
        else:
            # AI move
            ai_move = get_ai_move(board)
            board.push(ai_move)
            print(f"AI plays: {ai_move.uci()}")
            
    print("Game Over! Result:", board.result())
    
def is_endgame(board):
    """Improved endgame detection"""
    queens = board.pieces(chess.QUEEN, chess.WHITE) | board.pieces(chess.QUEEN, chess.BLACK)
    minors = len(board.pieces(chess.KNIGHT, chess.WHITE)) + \
             len(board.pieces(chess.BISHOP, chess.WHITE)) + \
             len(board.pieces(chess.KNIGHT, chess.BLACK)) + \
             len(board.pieces(chess.BISHOP, chess.BLACK))
    return len(queens) == 0 and minors <= 2
    
def evaluate_king_activity(board, color):
    """Reward king centralization in endgames."""
    if not is_endgame(board):
        return 0
    
    king_square = board.king(color)
    if not king_square:
        return 0
    
    target = chess.D4 if color == chess.WHITE else chess.D5
    distance = chess.square_distance(king_square, target)
    
    return (4 - distance) * 0.3

if __name__ == "__main__":
    play_chess()