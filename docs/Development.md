# Development

Documentation related to development and tooling.

## MLIR LSP

The MLIR project includes an LSP server implementation that provides editor integration for editing MLIR assembly (diagnostics, documentation, autocomplete, ..)[^1].
Because Dynamatic uses additional out-of-tree MLIR dialects (Dynamatic handshake, Dynamatic hw), we provide an extended version of this LSP server with these dialects registered.

This server is built automatically during the Dynamatic compilation flow, and can be found at `./bin/dynamatic-mlir-lsp-server` once ready. Usage of this LSP is IDE-specific.

### VSCode

**TODO**

### NeoVim (lspconfig)

NVIM's `lspconfig`[^2] provides integration for the normal MLIR lsp server. We recommend relying on this, and only conditionally overriding the `cmd` used to start the server if inside the Dynamatic folder hierarchy.

For example, this can be achieved by overriding the `cmd` of the LSP server when registering it:
```lua
lspconfig.mlir_lsp_server.setup({
    cmd = (function()
        local fallback = { "mlir_lsp_server" }

        local dynamatic_proj_path = vim.fs.find('dynamatic', { path = vim.fn.getcwd(), upward = true })[1]
        if not dynamatic_proj_path then return fallback end -- not in dynamatic

        local lsp_bin = dynamatic_proj_path .. "/bin/dynamatic-mlir-lsp-server"
        if not vim.uv.fs_stat(lsp_bin) then
            vim.notify("Dynamatic MLIR LSP does not exist.", vim.log.levels.WARN)
            return fallback
        end

        vim.notify("Using local MLIR LSP (" .. dynamatic_proj_path .. ")", vim.log.levels.INFO)
        return { lsp_bin }
    end)(),
    -- ...
})
```

Alternatively, you can add an `lspconfig` hook to override the server `cmd` during initialization. Note that this
hook must be registered *before* you use `lspconfig` to setup `mlir_lsp_server`.
```lua
lspconfig.util.on_setup = lspconfig.util.add_hook_before(lspconfig.util.on_setup, function(config)
    if config.name ~= "mlir_lsp_server" then return end -- other lsp

    local dynamatic_proj_path = vim.fs.find('dynamatic', { path = vim.fn.getcwd(), upward = true })[1]
    if not dynamatic_proj_path then return end -- not in dynamatic

    local lsp_bin = dynamatic_proj_path .. "/bin/dynamatic-mlir-lsp-server"
    if not vim.uv.fs_stat(lsp_bin) then
        vim.notify("Dynamatic MLIR LSP does not exist.", vim.log.levels.WARN)
        return
    end

    vim.notify("Using local MLIR LSP (" .. dynamatic_proj_path .. ")", vim.log.levels.INFO)
    config.cmd = { lsp_bin }
end)
lspconfig.mlir_lsp_server.setup({
    -- ...
})
```

[^1]: https://mlir.llvm.org/docs/Tools/MLIRLSP/ 
[^2]: https://github.com/neovim/nvim-lspconfig
