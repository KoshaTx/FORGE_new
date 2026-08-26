-- Pandoc filter for the ICLR-format build of the FORGE manuscript.
--
-- The Nature Biotechnology build has its own filter because that template owns a bespoke
-- front-matter panel. Here the requirements are different and simpler: the title block is
-- anonymous, the abstract is set by the template rather than appearing as a section, and
-- ICLR numbers its sections, so the level-1 headers stay as ordinary sections.

local stringify = (require 'pandoc.utils').stringify

local function starts_with(text, prefix)
  return text:sub(1, #prefix) == prefix
end

-- Long SMILES strings are unbreakable and overflow narrow table columns. Escaping is done
-- character by character: chaining gsubs escapes an underscore to \_ and then rewrites the
-- backslash it just inserted, which leaves a bare underscore and a maths-mode error.
local LATEX_ESCAPE = {
  ['\\'] = '\\textbackslash{}', ['{'] = '\\{', ['}'] = '\\}', ['#'] = '\\#',
  ['%'] = '\\%', ['$'] = '\\$', ['&'] = '\\&', ['_'] = '\\_',
  ['~'] = '\\textasciitilde{}', ['^'] = '\\textasciicircum{}',
}

-- The journal build carries the animal work; blocks marked for it are dropped here.
function Div(element)
  if element.classes:includes('nature-only') then
    return {}
  end
  if element.classes:includes('iclr-only') then
    return element.content
  end
  return nil
end

function Code(el)
  if #el.text < 9 then
    return nil
  end
  local pieces = {}
  for index = 1, #el.text do
    local char = el.text:sub(index, index)
    pieces[#pieces + 1] = LATEX_ESCAPE[char] or char
    if index % 4 == 0 and index < #el.text then
      pieces[#pieces + 1] = '\\allowbreak{}'
    end
  end
  return pandoc.RawInline('latex', '\\texttt{\\footnotesize ' .. table.concat(pieces) .. '}')
end

-- The ICLR style files are written for pdflatex, whose default encoding has no Greek.
-- The manuscript uses exactly two such characters, in "alpha,omega-diol".
local GREEK = { ['\u{3b1}'] = '\\ensuremath{\\alpha}', ['\u{3c9}'] = '\\ensuremath{\\omega}' }

function Str(element)
  local replacement = GREEK[element.text]
  if replacement then
    return pandoc.RawInline('latex', replacement)
  end
  if element.text:find('\u{3b1}') or element.text:find('\u{3c9}') then
    local pieces = {}
    for _, char in utf8.codes(element.text) do
      local ch = utf8.char(char)
      pieces[#pieces + 1] = GREEK[ch] or ch
    end
    return pandoc.RawInline('latex', table.concat(pieces))
  end
  return nil
end


-- No Image handler here on purpose. Pandoc's default emission is
--   width=\linewidth,height=\textheight,keepaspectratio
-- which bounds a figure without distorting it. Setting a height attribute makes pandoc
-- honour both dimensions literally and drop keepaspectratio, which stretched every figure
-- vertically. The oversized-float problem that once prompted such an override is gone now
-- that the figures are authored at print size rather than at fourteen inches wide.

function Pandoc(doc)
  local out = {}
  local abstract_blocks = {}
  local capture_abstract = false
  local captured = false
  local seen_title = false
  local front_matter_taken = 0

  for _, block in ipairs(doc.blocks) do
    local text = stringify(block)

    if block.t == 'Header' and block.level == 1 and not seen_title then
      -- The first level-1 header is the paper title; the template sets it.
      seen_title = true
      doc.meta.title = pandoc.MetaInlines(block.content)
    elseif seen_title and front_matter_taken < 2 and
           (block.t == 'Para' or block.t == 'Plain') then
      -- The author and affiliation paragraphs are dropped: this build is anonymous.
      front_matter_taken = front_matter_taken + 1
    elseif block.t == 'Header' and text == 'Abstract' then
      capture_abstract = true
    elseif capture_abstract and not captured and
           (block.t == 'Para' or block.t == 'Plain') then
      table.insert(abstract_blocks, block)
      captured = true
      capture_abstract = false
    elseif block.t == 'BlockQuote' and starts_with(text, 'No figure artwork') then
      -- An internal production note, not part of the paper.
    else
      table.insert(out, block)
    end
  end

  doc.blocks = out
  doc.meta.abstract = pandoc.MetaBlocks(abstract_blocks)
  return doc
end
