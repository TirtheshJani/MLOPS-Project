import axios from  axios;

const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL || http://localhost:8000 });

export const summarize = async (text) => {
  const { data } = await api.post(/summarize, { text });
  return data; // { summary: string }
};

export default api;
