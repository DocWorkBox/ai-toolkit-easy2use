import net from 'net';

// Retry on the same server, without probing and releasing a candidate port.
export async function listenPublic(server: net.Server, preferred: number, fallback: boolean): Promise<number> {
  const bind = (port: number) => new Promise<number>((resolve, reject) => {
    const failed = (error: Error) => { server.off('listening', ready); reject(error); };
    const ready = () => {
      server.off('error', failed);
      resolve((server.address() as net.AddressInfo).port);
    };
    server.once('error', failed).once('listening', ready).listen(port);
  });
  try {
    return await bind(preferred);
  } catch (err) {
    if (!fallback || !preferred || !['EACCES', 'EADDRINUSE'].includes((err as NodeJS.ErrnoException).code)) throw err;
    return bind(0);
  }
}
