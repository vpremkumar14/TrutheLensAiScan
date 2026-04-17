import { useState } from 'react'
import { motion } from 'framer-motion'
import FileUpload from '../components/FileUpload'
import ResultCard from '../components/ResultCard'
import LoadingSpinner from '../components/LoadingSpinner'
import { detectImage } from '../utils/api'
import robotBg from '../assets/hero-bg.png'

const ImageDetection = () => {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileSelect = async (selectedFile) => {
    if (!selectedFile) return

    setFile(selectedFile)
    setError(null)
    setResult(null)

    // Create preview
    const reader = new FileReader()
    reader.onloadend = () => {
      setPreview(reader.result)
    }
    reader.readAsDataURL(selectedFile)

    // Send to backend
    setLoading(true)
    try {
      const data = await detectImage(selectedFile)
      setResult(data)
    } catch (err) {
      setError(err.message || 'Error analyzing image. Please try again.')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-dark-900 py-12 px-4 md:px-8 relative">
      {/* Background Layer */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-r from-dark-900 via-dark-800 to-dark-900"></div>
        <div 
          className="absolute inset-0 opacity-40"
          style={{
            backgroundImage: `url(${robotBg})`,
            backgroundPosition: 'right center',
            backgroundRepeat: 'no-repeat',
            backgroundSize: 'auto 100%',
            backgroundAttachment: 'fixed',
          }}
        ></div>
        <div className="absolute inset-0 opacity-20" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 800'%3E%3Cline x1='0' y1='0' x2='1200' y2='800' stroke='%2306b6d4' stroke-width='0.5' opacity='0.3'/%3E%3Cline x1='1200' y1='0' x2='0' y2='800' stroke='%2306b6d4' stroke-width='0.5' opacity='0.3'/%3E%3C/svg%3E")`,
        }}></div>
        <div className="absolute top-1/4 right-0 w-96 h-96 bg-cyan-500 rounded-full mix-blend-screen filter blur-3xl opacity-10 animate-blob"></div>
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-cyan-400 rounded-full mix-blend-screen filter blur-3xl opacity-5 animate-blob"></div>
      </div>
      <div className="max-w-6xl mx-auto relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">Image Detection</span>
          </h1>
          <p className="text-gray-400 text-lg">
            Upload an image to detect if it's AI-generated or manipulated
          </p>
        </motion.div>

        {/* Upload Section */}
        <div className="mb-12">
          <FileUpload
            onFileSelect={handleFileSelect}
            accept="image/*"
            loading={loading}
          />
        </div>

        {/* File Info */}
        {file && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass rounded-xl p-6 mb-8"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="w-16 h-16 rounded-lg bg-dark-700 flex items-center justify-center">
                  <svg className="w-8 h-8 text-cyan-400" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4z" />
                  </svg>
                </div>
                <div>
                  <p className="font-semibold">{file.name}</p>
                  <p className="text-sm text-gray-400">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
              <button
                onClick={() => {
                  setFile(null)
                  setPreview(null)
                  setResult(null)
                }}
                className="text-red-400 hover:text-red-300 transition"
              >
                Clear
              </button>
            </div>
          </motion.div>
        )}

        {/* Loading State */}
        {loading && <LoadingSpinner />}

        {/* Error State */}
        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass rounded-xl p-6 border border-red-500/30 bg-red-500/10 mb-8"
          >
            <p className="text-red-400">⚠ {error}</p>
          </motion.div>
        )}

        {/* Result */}
        {result && !loading && (
          <ResultCard
            prediction={result.prediction}
            confidence={result.confidence}
            explanation={result.explanation}
            imageUrl={preview}
            type="image"
          />
        )}
      </div>
    </div>
  )
}

export default ImageDetection
