import axios from "axios"

const API = axios.create({
  baseURL: "http://54.198.216.48:8000/"
})

export const predictFile = (file) => {
  const fd = new FormData()
  fd.append("file", file)
  return API.post("/predict", fd, {
    headers: { "Content-Type": "multipart/form-data" }
  })
}

export default API
