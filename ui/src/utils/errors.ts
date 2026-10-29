export function getApiErrorMessage(error: any): string {
  const detail = error?.response?.data?.error;
  const message = typeof detail === 'string' ? detail : error?.message || '无法连接服务，请检查服务是否仍在运行';
  const status = error?.response?.status;
  return status ? `${message} (HTTP ${status})` : message;
}
