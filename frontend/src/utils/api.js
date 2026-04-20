import axios from 'axios'

// ✅ Configured for dynamic deployment and local fallbacks
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000"

// ✅ Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
})

// =========================
// IMAGE DETECTION
// =========================
export const detectImage = async (file) => {
  try {
    const formData = new FormData()
    formData.append('file', file)

    const response = await apiClient.post('/api/detect-image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    // ✅ Return directly (no success field)
    return response.data

  } catch (error) {
    console.error("Image Error:", error)
    throw new Error("Image detection failed")
  }
}

// =========================
// VIDEO DETECTION
// =========================
export const detectVideo = async (file) => {
  try {
    const formData = new FormData()
    formData.append('file', file)

    const response = await apiClient.post('/api/detect-video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    return response.data

  } catch (error) {
    console.error("Video Error:", error)
    throw new Error("Video detection failed")
  }
}

// =========================
// HEALTH CHECK
// =========================
export const checkHealth = async () => {
  try {
    const response = await apiClient.get('/api/health')
    return response.data
  } catch (error) {
    throw new Error("Backend not connected")
  }
}