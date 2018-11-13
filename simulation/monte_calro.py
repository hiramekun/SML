import numpy as np

N_REPEAT = 100000


def is_ended(board, flag):
    if board[0, 0] == flag:
        if board[0, 1] == flag & board[0, 2] == flag:
            return True
        elif board[1, 0] == flag & board[2, 0] == flag:
            return True
        elif board[1, 1] == flag & board[2, 2] == flag:
            return True

    elif board[1, 1] == flag:
        if board[0, 1] == flag & board[2, 1] == flag:
            return True
        elif board[1, 0] == flag & board[1, 2] == flag:
            return True
        elif board[0, 2] == flag & board[2, 0] == flag:
            return True

    elif board[2, 2] == flag:
        if board[2, 1] == flag & board[2, 0] == flag:
            return True
        elif board[1, 2] == flag & board[0, 2] == flag:
            return True

    return False


def do_game():
    win_a = 0
    win_b = 0
    for i in range(N_REPEAT):
        if i % 1000 == 0:
            print(i)  # for debug

        board = np.zeros((3, 3), dtype='int')
        xys = []
        for j in range(3):
            for k in range(3):
                xys.append((j, k))
        xys = np.array(xys)
        a_turn = True
        while len(xys) != 0:
            length = len(xys)
            idx = np.random.choice(length) - 1
            n_x = xys[idx][0]
            n_y = xys[idx][1]
            xys = np.delete(xys, np.where(idx), axis=0)
            if a_turn:
                board[n_x, n_y] = 1
                if is_ended(board, 1):
                    win_a += 1
                    break
            else:
                board[n_x, n_y] = -1
                if is_ended(board, -1):
                    win_b += 1
                    break
            a_turn = not a_turn

    print()
    print(f'先手の勝つ確率: {win_a / N_REPEAT}')
    print(f'後手の勝つ確率: {win_b / N_REPEAT}')
    std_a = np.std(np.hstack([np.repeat(1, win_a), np.repeat(0, N_REPEAT - win_a)]), ddof=1) / N_REPEAT ** (1 / 2)
    std_b = np.std(np.hstack([np.repeat(1, win_b), np.repeat(0, N_REPEAT - win_b)]), ddof=1) / N_REPEAT ** (1 / 2)
    print(f'先手の勝つ確率の標準誤差: {std_a}')
    print(f'後手の勝つ確率の標準誤差: {std_b}')


if __name__ == '__main__':
    do_game()
