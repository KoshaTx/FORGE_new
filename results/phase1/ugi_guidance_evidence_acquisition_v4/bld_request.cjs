// Run the supplier's unchanged public request formatter without filesystem/network bindings.
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const bd = process.argv[2];
if (!['BD305903', 'BD57579', 'BD151911'].includes(bd)) throw new Error('Unlisted product');
const context = vm.createContext({setInterval: () => 0, setTimeout: () => 0});
for (const filename of ['bld_hex.js', 'bld_js.js']) {
  vm.runInContext(fs.readFileSync(path.join(__dirname, 'batch_03', filename), 'utf8'), context,
                  {timeout: 1000});
}
context.bd = bd;
const body = vm.runInContext(`
  var $ = {each: function(obj, fn) {
    for (var key in obj) if (fn.call(obj[key], key, obj[key]) === false) break;
  }};
  var post_obj = {timestamp: new Date().getTime(), bd: bd};
  post_obj['_'] = getSign(post_obj, 1);
  post_obj['__'] = ['timestamp', 'bd'];
  JSON.stringify(post_obj);
`, context, {timeout: 1000});
process.stdout.write(JSON.stringify({params: Buffer.from(body, 'utf8').toString('base64')}));
