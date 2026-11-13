const { test } = require('node:test');
const assert = require('node:assert/strict');
const net = require('node:net');
const { listenPublic } = require('../ui/dist/cron/publicPort.js');

test('prefer requested port, only switch when actual bind fails', async () => {
  const first = net.createServer(), second = net.createServer(), third = net.createServer();
  try {
    const preferred = await listenPublic(first, 0, false);
    await assert.rejects(listenPublic(second, preferred, false), {code: 'EADDRINUSE'});
    const fallback = await listenPublic(second, preferred, true);
    assert.notEqual(fallback, preferred);
    assert.equal(second.address().port, fallback);
    await new Promise(resolve => first.close(resolve));
    assert.equal(await listenPublic(third, preferred, true), preferred);
  } finally {
    for (const server of [first, second, third]) if (server.listening) server.close();
  }
});
