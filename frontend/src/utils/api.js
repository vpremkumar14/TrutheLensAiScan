import axios from 'axios'

// Update this to your backend URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api'

// Create axios instance with timeout
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 second timeout for large files
})

/**
 * Detect if image is real or fake
 * @param {File} file - Image file to analyze
 * @returns {Promise<Object>} - Prediction result with confidence
 */
export const detectImage = async (file) => {
  try {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await apiClient.post('/detect/image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Detection failed')
    }
    
    return response.data
  } catch (error) {
    throw new Error(error.response?.data?.error || error.message || 'Image detection failed')
  }
}

/**
 * Detect if video is real or fake
 * @param {File} file - Video file to analyze
 * @returns {Promise<Object>} - Prediction result with confidence
 */
export const detectVideo = async (file) => {
  try {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await apiClient.post('/detect/video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Detection failed')
    }
    
    return response.data
  } catch (error) {
    throw new Error(error.response?.data?.error || error.message || 'Video detection failed')
  }
}

/**
 * Detect from image URL
 * @param {string} url - Image URL to analyze
 * @returns {Promise<Object>} - Prediction result with confidence
 */
export const detectFromUrl = async (url) => {
  try {
    const response = await apiClient.post('/detect/url', { url })
    
    if (!response.data.success) {
      throw new Error(response.data.error || 'Detection failed')
    }
    
    return response.data
  } catch (error) {
    throw new Error(error.response?.data?.error || error.message || 'URL detection failed')
  }
}

/**
 * Health check
 * @returns {Promise<Object>} - API health status
 */
export const checkHealth = async () => {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error) {
    throw new Error('API connection failed')
  }
}

/**
 * Get API info
 * @returns {Promise<Object>} - API information
 */
export const getApiInfo = async () => {
  try {
    const response = await apiClient.get('/info')
    return response.data
  } catch (error) {
    throw new Error('Failed to fetch API info')
  }
}
