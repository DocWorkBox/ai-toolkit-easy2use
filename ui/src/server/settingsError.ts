// Prisma errors can embed the entire query, including HF_TOKEN. Classify them
// without returning or logging that query or the request body.
export function settingsError(error: any): string {
  const message = [error?.message, error?.meta?.message, error?.cause?.message].filter(Boolean).join(' ');
  const code = String(error?.code || error?.errorCode || 'UNKNOWN');
  const safeCode = /^[A-Z0-9_]+$/.test(code) ? code : 'UNKNOWN';
  let reason: string;
  if (/readonly|read.only/i.test(message)) {
    reason = '数据库为只读，检查 aitk_db.db 及其所在目录的写入权限';
  } else if (/locked|busy|timed? ?out|timeout/i.test(message) || ['P1008', 'P2024', 'P2034'].includes(code)) {
    reason = '数据库被占用或操作超时，请稍后重试；检查是否同时运行了多个训练器服务';
  } else if (/disk.*full|SQLITE_FULL/i.test(message)) {
    reason = '数据库所在磁盘空间不足';
  } else if (/unable to open|permission|access denied/i.test(message) || ['P1003', 'P1010'].includes(code)) {
    reason = '无法打开数据库，检查 aitk_db.db 路径及目录访问权限';
  } else if (/no such (table|column)/i.test(message) || ['P2021', 'P2022'].includes(code)) {
    reason = '数据库结构与程序不一致，请通过启动器完成数据库更新';
  } else if (/malformed|corrupt/i.test(message)) {
    reason = '数据库文件损坏，停止写入并检查数据库文件';
  } else if (['P2002', 'P2003', 'P2004'].includes(code)) {
    reason = '数据库约束检查失败，本次设置未写入';
  } else {
    reason = '数据库操作失败，设置未保存';
  }
  return `${reason} (${safeCode})`;
}
