## start the web server

# python -m uvicorn sft_fastapi:app --host 0.0.0.0 --port 8000

## send request to the server

# curl -G "http://localhost:8000/recommend" \
#   --data-urlencode "prompt=$(cat prompt.txt)" \
#   --data-urlencode "top_k=3"

curl http://localhost:8000/health

curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "sequent": "{1}   FORALL (A, B: simple_polygon_2d, j: below(A`num_vertices), i: nat): LET IV = injected_vertices(A, B, A`num_vertices), s = edges_of_polygon(A)(j), L = injected_vertices(A, B, j)`length, Q = injected_edge_seq(s, injected_edge(s, B)) IN i < IV`length AND i >= L AND i < Q`length + L IMPLIES IV`seq(i) = Q`seq(i - L)",
    "prev_commands": ["None", "None", "None"],
    "top_k": 3
  }'