docker build -f Dockerfile.general -t timemagic/rl-mcp:general .

docker build -f Dockerfile.proxy-mcp -t timemagic/rl-mcp:proxy-mcp .

docker build -f Dockerfile.fetch-mcp -t timemagic/rl-mcp:fetch-mcp .