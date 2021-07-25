{
  let result = `#pragma once
#include <string>

`;

  for (let f of $.glob('../x64/Release/*.ptx')) {
    let s = $.readText(f).replace(/\/\/.*\r?\n/g, '').trim().replace(/\\/g, '\\\\').replace(/\r?\n/g, '\\n').replace(/\s+/g, ' ').replace(/"/g, '\\"');
    let p = s.match(/.{1,119}[^\\]/g);
    $.echo(p.length);
    result += `inline std::string ptx_program_${path.basename(f, '.cu.ptx')}(){ 
  std::string s;
  ${[].map.call(p, x => `s += "${x}";`).join('\n  ')}
  return s;
}\n\n`
  }

  fs.writeFileSync('ptx_programs.h', result);
}

{
  let result = `#pragma once
#include <string>

`;

  for (let f of $.glob('../x64/Release/*.fxo')) {
    let s = fs.readFileSync(f).toString('base64').replace(/\/\/.*\r?\n/g, '').trim().replace(/\\/g, '\\\\').replace(/\r?\n/g, '\\n').replace(/\s+/g, ' ').replace(/"/g, '\\"');
    let p = s.match(/.{1,119}[^\\]/g);
    $.echo(p.length);
    result += `inline std::string dx_shader_${path.basename(f, '.fxo')}(){ 
  std::string s;
  ${[].map.call(p, x => `s += "${x}";`).join('\n  ')}
  return s;
}\n\n`
  }

  fs.writeFileSync('dx_shaders.h', result);

}
