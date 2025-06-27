#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include "neuralnet.h"

#define WINDOW_WIDTH 300
#define WINDOW_HEIGHT 300
#define CELL_SIZE 100
#define GRID_SIZE 3

// Function prototype for SDL_RenderDrawCircle
void SDL_RenderDrawCircle(SDL_Renderer* renderer, int x0, int y0, int radius);

// 0 = empty, 1 = user (X), -1 = AI (O)
void draw_grid(SDL_Renderer* renderer, int* data) {
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            SDL_Rect cell = { x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE };
            SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
            SDL_RenderFillRect(renderer, &cell);
            SDL_SetRenderDrawColor(renderer, 80, 80, 80, 255);
            SDL_RenderDrawRect(renderer, &cell);
            int v = data[y * GRID_SIZE + x];
            if (v == 1) {
                // Draw X
                SDL_SetRenderDrawColor(renderer, 0, 200, 0, 255);
                SDL_RenderDrawLine(renderer, cell.x+10, cell.y+10, cell.x+CELL_SIZE-10, cell.y+CELL_SIZE-10);
                SDL_RenderDrawLine(renderer, cell.x+CELL_SIZE-10, cell.y+10, cell.x+10, cell.y+CELL_SIZE-10);
            } else if (v == -1) {
                // Draw O
                SDL_SetRenderDrawColor(renderer, 200, 0, 0, 255);
                for (int r = 0; r < CELL_SIZE/2-10; r+=2) {
                    SDL_RenderDrawCircle(renderer, cell.x+CELL_SIZE/2, cell.y+CELL_SIZE/2, r);
                }
            }
        }
    }
}

// Helper to draw a circle (Bresenham's algorithm)
void SDL_RenderDrawCircle(SDL_Renderer* renderer, int x0, int y0, int radius) {
    int x = radius-1;
    int y = 0;
    int dx = 1;
    int dy = 1;
    int err = dx - (radius << 1);
    while (x >= y) {
        SDL_RenderDrawPoint(renderer, x0 + x, y0 + y);
        SDL_RenderDrawPoint(renderer, x0 + y, y0 + x);
        SDL_RenderDrawPoint(renderer, x0 - y, y0 + x);
        SDL_RenderDrawPoint(renderer, x0 - x, y0 + y);
        SDL_RenderDrawPoint(renderer, x0 - x, y0 - y);
        SDL_RenderDrawPoint(renderer, x0 - y, y0 - x);
        SDL_RenderDrawPoint(renderer, x0 + y, y0 - x);
        SDL_RenderDrawPoint(renderer, x0 + x, y0 - y);
        if (err <= 0) {
            y++;
            err += dy;
            dy += 2;
        }
        if (err > 0) {
            x--;
            dx += 2;
            err += dx - (radius << 1);
        }
    }
}

// Check for win: returns 1 (user), -1 (AI), 0 (no win)
int check_win(int* grid) {
    for (int i = 0; i < GRID_SIZE; i++) {
        int row = grid[i*GRID_SIZE] + grid[i*GRID_SIZE+1] + grid[i*GRID_SIZE+2];
        int col = grid[i] + grid[i+3] + grid[i+6];
        if (row == 3 || col == 3) return 1;
        if (row == -3 || col == -3) return -1;
    }
    int diag1 = grid[0] + grid[4] + grid[8];
    int diag2 = grid[2] + grid[4] + grid[6];
    if (diag1 == 3 || diag2 == 3) return 1;
    if (diag1 == -3 || diag2 == -3) return -1;
    return 0;
}

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window* window = SDL_CreateWindow("Tic-Tac-Toe AI", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    int running = 1;
    int grid[GRID_SIZE * GRID_SIZE] = {0};
    int turn = 1; // 1 = user, -1 = AI
    // --- Neural network setup ---
    NeuralNet* nn = nn_create(9, 18, 9, 0.1f);

    // --- Self-play training loop ---
    int train_games = 10000;
    for (int game = 0; game < train_games; game++) {
        int board[9] = {0};
        int player = 1; // 1 = X, -1 = O
        int moves = 0;
        int winner = 0;
        while (winner == 0 && moves < 9) {
            // Prepare input
            Matrix* input = matrix_create(9, 1);
            for (int i = 0; i < 9; i++) input->data[i] = (float)board[i];
            Matrix* output = nn_predict(nn, input);
            // Pick best move
            float best = -1e9;
            int best_idx = -1;
            for (int i = 0; i < 9; i++) {
                if (board[i] == 0 && output->data[i] > best) {
                    best = output->data[i];
                    best_idx = i;
                }
            }
            if (best_idx == -1) break;
            board[best_idx] = player;
            moves++;
            winner = check_win(board);
            // Prepare target: reward winning move, penalize losing
            Matrix* target = matrix_create(9, 1);
            for (int i = 0; i < 9; i++) target->data[i] = output->data[i];
            if (winner == player) {
                target->data[best_idx] = 1.0f; // Reward
            } else if (winner == -player) {
                target->data[best_idx] = 0.0f; // Penalize
            } else {
                target->data[best_idx] = 0.5f; // Neutral
            }
            nn_train(nn, input, target);
            matrix_free(input);
            matrix_free(output);
            matrix_free(target);
            player = -player;
        }
    }

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = 0;
            if (e.type == SDL_MOUSEBUTTONDOWN && turn == 1) {
                int mx = e.button.x / CELL_SIZE;
                int my = e.button.y / CELL_SIZE;
                int idx = my * GRID_SIZE + mx;
                if (mx >= 0 && mx < GRID_SIZE && my >= 0 && my < GRID_SIZE && grid[idx] == 0) {
                    grid[idx] = 1;
                    turn = -1;
                }
            }
        }
        // --- AI move using neural network ---
        if (turn == -1) {
            // Prepare input
            Matrix* input = matrix_create(9, 1);
            for (int i = 0; i < 9; i++) input->data[i] = (float)grid[i];
            Matrix* output = nn_predict(nn, input);
            // Find best move (highest output for empty cell)
            float best = -1e9;
            int best_idx = -1;
            for (int i = 0; i < 9; i++) {
                if (grid[i] == 0 && output->data[i] > best) {
                    best = output->data[i];
                    best_idx = i;
                }
            }
            if (best_idx != -1) {
                grid[best_idx] = -1;
                turn = 1;
            }
            matrix_free(input);
            matrix_free(output);
        }
        int winner = check_win(grid);
        int is_draw = 1;
        for (int i = 0; i < 9; i++) {
            if (grid[i] == 0) {
                is_draw = 0;
                break;
            }
        }
        if (winner != 0) {
            printf("%s wins!\n", winner == 1 ? "User" : "AI");
            SDL_Delay(1000);
            for (int i = 0; i < 9; i++) grid[i] = 0;
            turn = 1;
        } else if (is_draw) {
            printf("Draw!\n");
            SDL_Delay(1000);
            for (int i = 0; i < 9; i++) grid[i] = 0;
            turn = 1;
        }
        SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
        SDL_RenderClear(renderer);
        draw_grid(renderer, grid);
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }
    nn_free(nn);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

#ifdef _WIN32
#include <windows.h>
int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    return main(__argc, __argv);
}
#endif
