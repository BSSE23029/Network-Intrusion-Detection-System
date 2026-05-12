import axios from "axios"

const API = axios.create({
  baseURL: "http://3.84.61.149:8000"
})

export const predictFile = (file) => {
  const fd = new FormData()
  fd.append("file", file)
  return API.post("/predict", fd, {
    headers: { "Content-Type": "multipart/form-data" }
  })
}

export default API
