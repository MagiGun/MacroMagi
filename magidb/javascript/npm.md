---
title: npm and package.json
summary: npm install installs what package.json lists; npm install name adds one; npm run name runs a script; npm test runs the tests.
tags: [javascript, npm, node, packages]
---
# npm and package.json

```
npm install              # everything package.json lists, into node_modules
npm install lodash       # add one
npm install -D vitest    # add one only for development
npm run build            # the "build" script of package.json
npm test
```

`Cannot find module 'x'`: run `npm install`. Never edit `node_modules`; it is rebuilt from package.json.
