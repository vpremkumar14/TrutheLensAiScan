echo "🚀 Starting TruthLens Live AI Server..."

# Start Backend in Background
echo "Starting PyTorch AI Backend on port 5000..."
cd backend
start /B python app.py
cd ..

# Wait a moment for backend to warm up
Start-Sleep -Seconds 5

# Start Frontend in Background
echo "Starting React Frontend on port 3000..."
cd frontend
start /B npm run dev
cd ..

Start-Sleep -Seconds 3

# Deploy to the Internet using LocalTunnel!
echo "DEPLOYING TO THE INTERNET..."
echo "Your live internet link will appear below within seconds:"
npx localtunnel --port 3000
