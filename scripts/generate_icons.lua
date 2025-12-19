---@param a number
---@param b number
---@param t number
---@return number
local function lerp(a, b, t)
    return a + (b - a) * t
end

---@param a number[]
---@param b number[]
---@param t number
---@return number[]
local function lerp_color(a, b, t)
    return {
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
        lerp(a[3], b[3], t)
    }
end

---@param width integer
---@param height integer
---@return string
local function generate_rgba(width, height)
    ---@type number[]
    local top_left     = { 0.45, 0.20, 0.75 }
    ---@type number[]
    local top_right    = { 0.20, 0.85, 0.85 }
    ---@type number[]
    local bottom_left  = { 0.95, 0.30, 0.65 }
    ---@type number[]
    local bottom_right = { 0.30, 0.85, 0.65 }

    ---@type string[]
    local out          = {}

    for y = 0, height - 1 do
        local v     = y / (height - 1)
        local left  = lerp_color(top_left, bottom_left, v)
        local right = lerp_color(top_right, bottom_right, v)

        for x = 0, width - 1 do
            local u = x / (width - 1)
            local c = lerp_color(left, right, u)

            out[#out + 1] = string.char(
                math.floor(c[1] * 255),
                math.floor(c[2] * 255),
                math.floor(c[3] * 255),
                255
            )
        end
    end

    return table.concat(out)
end

---@param path string
---@param data string
local function write_file(path, data)
    ---@type file
    local f = assert(io.open(path, "wb"))
    f:write(data)
    f:close()
end

---@param path string
local function ensure_dir(path)
    local is_windows = package.config:sub(1, 1) == "\\"
    if is_windows then
        os.execute('mkdir "' .. path .. '" 2>nul')
    else
        os.execute('mkdir -p "' .. path .. '"')
    end
end

ensure_dir("assets")
ensure_dir("assets/engine")

write_file("assets/engine/icon_256.rgba", generate_rgba(256, 256))
write_file("assets/engine/icon_32.rgba", generate_rgba(32, 32))

print("Generated vaporwave engine icons")
