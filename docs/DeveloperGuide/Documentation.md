# Documentation

Dynamatic's documentation is written in markdown, which is located in the `./docs` folder.

It is rendered to an HTML web page using [`mdbook`](https://rust-lang.github.io/mdBook/index.html), which is hosted at <https://epfl-lap.github.io/dynamatic/>, automatically on every push to the main repository.

## Compiling the Documentation

To render and view the documentation locally, please install [`mdbook`](https://github.com/rust-lang/mdBook), and the [`mdbook-alerts`](https://github.com/lambdalisue/rs-mdbook-alerts) plugin

Optionally, you can install the [`mdbook-linkcheck`](https://github.com/Michael-F-Bryan/mdbook-linkcheck) backend, to check for broken links in the documentation.

Then, from the root of the repository run:

- `mdbook build`: to compile the documentation to HTML.
- `mdbook serve`: to compile the documentation and host it on a local webserver. Navigate to the shown location (usually <localhost:3000>) to view the docs. The docs are automatically re-compiled when they are modified.

## Adding a new page

The structure of the documentation page is determined by the `./docs/SUMMARY.md` file.

If you add a new page, you must also list it in this file for it to show up.

Note that we try to mirror the documentation file structure in the `./docs` folder and the actual documentation structure.
