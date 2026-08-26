local stringify = pandoc.utils.stringify

local function starts_with(text, prefix)
  return text:sub(1, #prefix) == prefix
end

local function is_standalone_placeholder(text)
  return starts_with(text, "[RESULT PLACEHOLDER") or
         starts_with(text, "[METHOD PLACEHOLDER") or
         starts_with(text, "[PLACEHOLDER")
end

-- SMILES strings are long, unbreakable and sit in narrow supplementary table columns,
-- where they overflow into the neighbouring cell. seqsplit is not in the TeX Live basic
-- scheme, so break points are inserted directly. Escaping is done here rather than left
-- to pandoc because the raw inline bypasses its escaping.
local LATEX_ESCAPE = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}", ["#"] = "\\#",
  ["%"] = "\\%", ["$"] = "\\$", ["&"] = "\\&", ["_"] = "\\_",
  ["~"] = "\\textasciitilde{}", ["^"] = "\\textasciicircum{}",
}

-- The ICLR build stops at the cellular screen; blocks marked for it are dropped here.
function Div(element)
  if element.classes:includes('iclr-only') then
    return {}
  end
  if element.classes:includes('nature-only') then
    return element.content
  end
  return nil
end

function Code(el)
  if #el.text < 14 then
    return nil
  end
  local pieces = {}
  for index = 1, #el.text do
    local char = el.text:sub(index, index)
    pieces[#pieces + 1] = LATEX_ESCAPE[char] or char
    -- A break opportunity every few characters keeps a long string inside its column
    -- without hyphens, which would be ambiguous in a chemical string.
    if index % 5 == 0 and index < #el.text then
      pieces[#pieces + 1] = "\\allowbreak{}"
    end
  end
  return pandoc.RawInline("latex", "\\texttt{\\footnotesize " .. table.concat(pieces) .. "}")
end

function Image(image)
  if image.src:match("^manuscript/figures/figure[12]/") then
    image.attributes.width = "112%"
    return {
      pandoc.RawInline("latex", "\\makebox[\\linewidth][c]{"),
      image,
      pandoc.RawInline("latex", "}")
    }
  end
  return image
end

function Pandoc(doc)
  local out = {}
  local abstract_blocks = {}
  local author_blocks = {}
  local affiliation_blocks = {}
  local capture_abstract = false
  local captured_abstract_paragraph = false
  local seen_title = false
  local front_matter_taken = 0
  local references_open = false

  for _, block in ipairs(doc.blocks) do
    local text = stringify(block)

    if block.t == "Header" and block.level == 1 and not seen_title then
      -- The first level-1 header is the title whatever it says, and the LaTeX template
      -- owns the title block. Matching on the wording instead would drop a renamed
      -- title into the body as an ordinary section heading.
      seen_title = true
    elseif seen_title and front_matter_taken < 2 and
           (block.t == "Para" or block.t == "Plain") then
      -- The two paragraphs after the title are the author list and the affiliation
      -- block. They are captured into metadata so the opening panel in the template
      -- stays the single place that lays out the front matter, while the markdown
      -- remains the single place the names are edited.
      front_matter_taken = front_matter_taken + 1
      if front_matter_taken == 1 then
        table.insert(author_blocks, block)
      else
        table.insert(affiliation_blocks, block)
      end
    elseif block.t == "BlockQuote" and starts_with(text, "Draft status.") then
      -- The LaTeX template carries a compact working-draft note.
    elseif block.t == "Header" and text == "Abstract" then
      capture_abstract = true
    elseif capture_abstract and not captured_abstract_paragraph and
           (block.t == "Para" or block.t == "Plain") then
      table.insert(abstract_blocks, block)
      captured_abstract_paragraph = true
      capture_abstract = false
    elseif block.t == "Header" and text == "Figure legends" then
      table.insert(out, pandoc.RawBlock("latex", "\\clearpage"))
      table.insert(out, block)
    elseif block.t == "Header" and text == "References" then
      table.insert(out, pandoc.RawBlock("latex", "\\clearpage"))
      table.insert(out, block)
      table.insert(out, pandoc.RawBlock("latex", "\\begin{singlespace}"))
      references_open = true
    elseif (block.t == "Para" or block.t == "Plain") and
           is_standalone_placeholder(text) then
      table.insert(out, pandoc.RawBlock("latex", "\\begin{forgeplaceholdertext}"))
      table.insert(out, block)
      table.insert(out, pandoc.RawBlock("latex", "\\end{forgeplaceholdertext}"))
    else
      table.insert(out, block)
    end
  end

  if references_open then
    table.insert(out, pandoc.RawBlock("latex", "\\end{singlespace}"))
  end

  doc.blocks = out
  doc.meta.abstract = pandoc.MetaBlocks(abstract_blocks)
  doc.meta.authorline = pandoc.MetaBlocks(author_blocks)
  doc.meta.affiliationline = pandoc.MetaBlocks(affiliation_blocks)
  return doc
end
